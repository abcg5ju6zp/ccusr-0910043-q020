#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  rule_engine/engine/cache.py
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following disclaimer
#    in the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the project nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""按租户与编译上下文隔离的规则编译缓存管理器。

缓存层级为::

    CompiledRuleCache
        └── _TenantCache            （每个租户一个，互不可见）
                └── (规则文本, CompileOptions) -> 已发布结果 / 编译占位

设计要点：

* **隔离**：缓存键的第一部分是租户标识；租户之间没有任何共享槽位，一个
  租户批量提交唯一规则只会淘汰自己的条目，也不可能读到其它租户的对象。
* **语义安全的键**：:py:class:`CompileOptions` 覆盖 :py:class:`~rule_engine.Context`
  中每一个会改变编译结果语义的选项；不同解析上下文编译出的对象永不会互相命中。
* **单飞（single-flight）**：相同键的并发编译只有一个线程真正调用编译器，
  其余线程等待同一个可发布结果。
* **失败不污染**：编译失败时键上只删除占位、不留下任何对象，下一次相同请求
  会重新尝试编译；等待者收到协调者的同一个异常。
* **淘汰与评估并发安全**：被淘汰的只是缓存字典中的引用；已经拿到
  :py:class:`~rule_engine.Rule` 的评估线程持有自己的局部引用，评估期间对象不会
  被回收，且 AST 一经发布即不可变。
* **统计脱敏**：统计信息只包含计数与成本，进程级汇总不包含任何租户的规则文本。
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import threading
import time
from typing import Any, Callable, Mapping, NamedTuple, TYPE_CHECKING

from .. import __version__, errors

if TYPE_CHECKING:
    from ..builtins import BuiltinValueGenerator
    from .context import Context
    from .rule import Rule

__all__ = (
    'CacheConfig',
    'CacheStats',
    'CompiledRuleCache',
    'CompileOptions',
    'cache_key',
)

_CACHE_KEY_ATTR = '__rule_engine_cache_key__'
"""可调用对象通过该属性显式声明参与缓存键的稳定身份。"""

# 语法/语义可能随包版本变化，键中必须包含引擎版本。
_ENGINE_VERSION: tuple[int, ...] = tuple(int(part) for part in __version__.split('.') if part.isdigit())


def cache_key(key: Any) -> Callable[[Any], Any]:
    """装饰器：为函数或类声明一个参与缓存键的稳定身份。

    解析器（resolver / type_resolver）通常是闭包或每次新建的
    :py:class:`functools.partial`，缓存管理器无法从对象本身判断两个闭包是否
    语义等价。被装饰的对象会携带 *key*；两个使用相同 *key* 的解析器被视为
    同一编译上下文。未声明身份的可调用对象按对象身份区分，因此不会错误命中，
    只是无法跨实例共享缓存。
    """
    def decorator(obj: Any) -> Any:
        setattr(obj, _CACHE_KEY_ATTR, key)
        return obj
    return decorator


def _registered_key(obj: Any) -> Any:
    return getattr(obj, _CACHE_KEY_ATTR, None)


def _stable_key(value: Any, *, _depth: int = 0) -> Any:
    """把 Python 值规范化为可哈希、可比较的缓存键片段。

    对结构化值按内容比较；无法安全结构化的对象回退为对象身份。递归最多两层，
    避免对调用方提供的大型嵌套对象做无界遍历。
    """
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return ('scalar', value)
    if isinstance(value, (tuple, list, frozenset, set)) and _depth < 2:
        items = [_stable_key(item, _depth=_depth + 1) for item in value]
        with contextlib.suppress(TypeError):
            if isinstance(value, (frozenset, set)):
                return ('set', frozenset(items))
            return ('seq', tuple(items))
    if isinstance(value, Mapping) and _depth < 2:
        with contextlib.suppress(TypeError):
            return ('map', tuple(sorted(
                (_stable_key(k, _depth=_depth + 1), _stable_key(v, _depth=_depth + 1))
                for k, v in value.items()
            )))
    registered = _registered_key(value)
    if registered is not None:
        return ('registered', registered)
    try:
        hash(value)
    except TypeError:
        # 不可哈希且无法结构化的对象只能按身份区分——不会错误命中，只是无法共享。
        return ('identity', id(value))
    # Decimal / datetime / timedelta 以及其它定义了相等性的不可变标量按内容比较。
    return ('scalar', value)


def _fingerprint_timezone(tz: Any) -> Any:
    # tzutc/tzlocal 的不同实例按身份不相等，需要结构化比较。
    parts: list[Any] = [type(tz).__name__]
    for attr in ('_filename', '_name', '_offset'):  # tzfile / tzstr / tzoffset
        if hasattr(tz, attr):
            parts.append((attr, _stable_key(getattr(tz, attr))))
    return ('tz', tuple(parts))


def _fingerprint_decimal_context(ctx: Any) -> Any:
    # flags/caches 是运行期状态而非配置，不影响编译语义，不参与指纹。
    return ('decimal', (
        ctx.prec,
        ctx.rounding,
        ctx.Emin,
        ctx.Emax,
        ctx.capitals,
        ctx.clamp,
        frozenset(ctx.traps),
    ))


def _fingerprint_callable(value: Any) -> Any:
    from ..builtins import BuiltinValueGenerator
    # BuiltinValueGenerator 包装的是真正提供值的可调用对象。
    if isinstance(value, BuiltinValueGenerator):
        value = value.callable
    registered = _registered_key(value)
    if registered is not None:
        return ('callable', registered)
    if isinstance(value, functools.partial):
        # 解包 partial：默认的 re_groups 取值器每次构建都会生成新的 partial，
        # 但其语义只由被包装函数与参数决定。
        return ('partial', _fingerprint_callable(value.func),
                _stable_key(value.args), _stable_key(value.keywords))
    return ('callable', id(value))


def _fingerprint_builtins(builtins: Any) -> Any:
    # 由 Builtins 自己遍历内部取值，避免触发生成器（$now/$today）取值，
    # 也避免从外部访问其私有结构。
    return builtins._cache_fingerprint(_fingerprint_callable, _stable_key, _fingerprint_timezone)


class _CompositeKey(tuple):
    """(规则文本, 编译选项) 复合键。

    各分量一般已经可哈希（未声明身份的可调用对象回退为 ``id``）；这里再做一次
    防御性处理，使得调用方通过 :py:func:`cache_key` 注册了不可哈希值时，
    缓存仍按相等性正常工作，而不是抛出 :py:class:`TypeError`。
    """
    __slots__ = ()

    def __new__(cls, text: str, options: 'CompileOptions') -> '_CompositeKey':
        return super().__new__(cls, (text, options))

    def __hash__(self) -> int:  # type: ignore[override]
        digest = hash(self[0])
        for part in self[1]:
            try:
                part_hash = hash(part)
            except TypeError:
                part_hash = id(part)
            digest = (digest ^ part_hash) * 1000003 % (2 ** 63 - 1)
        return digest

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, tuple) or len(other) != 2:
            return False
        if self[0] != other[0]:
            return False
        for a, b in zip(self[1], other[1]):
            if a is b:
                continue
            try:
                equal = (a == b)
            except Exception:
                equal = False
            if not equal:
                return False
        return True


class CompileOptions(NamedTuple):
    """影响规则编译语义的完整选项集合。

    任何两个 :py:class:`~rule_engine.Context` 只要在本元组任一分量上不同，
    就必然落入不同的缓存槽位。自定义可调用对象（resolver / type_resolver）
    通过 :py:func:`cache_key` 声明稳定身份，未声明时按对象身份区分。
    """
    engine_version: tuple[int, ...]
    context_type: int
    regex_flags: int
    mapping_attribute_lookup: bool
    timezone: Any
    default_value: Any
    decimal_context: Any
    resolver: Any
    type_resolver: Any
    builtins: Any
    parser_identity: int

    @classmethod
    def from_context(cls, context: 'Context', parser: Any = None) -> 'CompileOptions':
        """项目内部接口说明。"""
        inputs = context._compile_fingerprint_inputs()
        if parser is None:
            from .rule import Rule
            parser = Rule.parser
        return cls(
            engine_version=_ENGINE_VERSION,
            context_type=id(type(context)),
            regex_flags=inputs['regex_flags'],
            mapping_attribute_lookup=inputs['mapping_attribute_lookup'],
            timezone=_fingerprint_timezone(inputs['default_timezone']),
            default_value=_stable_key(inputs['default_value']),
            decimal_context=_fingerprint_decimal_context(inputs['decimal_context']),
            resolver=_fingerprint_callable(inputs['resolver']),
            type_resolver=_fingerprint_callable(inputs['type_resolver']),
            builtins=_fingerprint_builtins(inputs['builtins']),
            parser_identity=id(parser),
        )


@dataclasses.dataclass(frozen=True)
class CacheConfig:
    """单个租户缓存的容量与生命周期配置。

    :param max_entries: 最多保存的已发布结果条目数；``0`` 表示不限。
        正在编译、尚未发布的占位不计入该数量。
    :param max_cost: 所有已发布结果的成本权重上限，``0`` 表示不限。
    :param default_cost: :py:attr:`cost_function` 为空时每个结果的固定权重。
    :param ttl: 结果存活秒数；``0`` 表示不过期。
    :param cost_function: 根据已编译规则计算成本权重的回调，必须返回非负整数。
    :param time_func: 计时函数（便于测试）。
    :param compile_wait_timeout: 等待同一键上其它线程编译完成的最长秒数。
    """
    max_entries: int = 128
    max_cost: int = 0
    default_cost: int = 1
    ttl: float = 0.0
    cost_function: Callable[['Rule'], int] | None = None
    time_func: Callable[[], float] = time.monotonic
    compile_wait_timeout: float = 30.0


@dataclasses.dataclass(frozen=True)
class CacheStats:
    """脱敏的缓存计数器快照，任何字段都不包含规则文本或上下文细节。"""
    tenants: int = 0
    current_entries: int = 0
    current_cost: int = 0
    hits: int = 0
    misses: int = 0
    published: int = 0
    compile_waits: int = 0
    compilation_failures: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0
    per_tenant: Mapping[str, 'CacheStats'] | None = None

    @property
    def hit_rate(self) -> float:
        """已发布结果的命中率；没有任何查找时为 ``0.0``。"""
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class _Entry(object):
    """已发布结果。AST 在此处一经写入即视为不可变。"""
    __slots__ = ('rule', 'cost', 'origin_context_id', 'created_at', 'expires_at', 'last_hit_at', 'hits')
    def __init__(self, rule: 'Rule', cost: int, origin_context_id: int, now: float, expires_at: float) -> None:
        self.rule = rule
        self.cost = cost
        self.origin_context_id = origin_context_id
        self.created_at = now
        self.expires_at = expires_at
        self.last_hit_at = now
        self.hits = 0

    def is_expired(self, now: float) -> bool:
        return self.expires_at > 0 and now >= self.expires_at


class _InFlight(object):
    """正在进行的编译占位，是单飞协调与结果发布的唯一同步点。"""
    __slots__ = ('event', 'rule', 'exc')
    def __init__(self) -> None:
        self.event = threading.Event()
        self.rule: 'Rule | None' = None
        self.exc: BaseException | None = None


class _TenantCache(object):
    """单个租户的隔离缓存：LRU + 成本权重 + TTL。"""
    def __init__(self, tenant_id: str, config: CacheConfig) -> None:
        self.tenant_id = tenant_id
        self.config = config
        self._lock = threading.RLock()
        self._entries: collections.OrderedDict[tuple[str, CompileOptions], Any] = collections.OrderedDict()
        self._cost = 0
        # 以下计数器仅在 self._lock 保护下读写。
        self.hits = 0
        self.misses = 0
        self.published = 0
        self.compile_waits = 0
        self.compilation_failures = 0
        self.evictions = 0
        self.expirations = 0
        self.invalidations = 0

    # -------------------------------------------------- 基础维护
    def _purge_expired(self, now: float) -> None:
        for key in list(self._entries):
            value = self._entries[key]
            if isinstance(value, _Entry) and value.is_expired(now):
                self._remove(key, value, expired=True)

    def _remove(self, key: tuple[str, CompileOptions], value: Any, *, expired: bool = False) -> None:
        existing = self._entries.pop(key, None)
        if existing is not value:
            # 键已被其它路径替换；成本/计数由替换方负责。
            return
        if isinstance(value, _Entry):
            self._cost -= value.cost
        if expired:
            self.expirations += 1

    def _stored_count(self) -> int:
        return sum(1 for value in self._entries.values() if isinstance(value, _Entry))

    def _enforce_limits(self) -> None:
        """淘汰最久未命中的已发布结果；绝不驱逐其它线程正在编译的占位。"""
        while self._entries:
            if self.config.max_entries <= 0 or self._stored_count() <= self.config.max_entries:
                if self.config.max_cost <= 0 or self._cost <= self.config.max_cost:
                    return
            victim_key = None
            # OrderedDict 从最旧向最新扫描，跳过编译占位。
            for key, value in self._entries.items():
                if isinstance(value, _Entry):
                    victim_key = key
                    break
            if victim_key is None:
                return
            value = self._entries.pop(victim_key)
            self._cost -= value.cost
            self.evictions += 1

    def _publish(self, key: tuple[str, CompileOptions], entry: _Entry) -> None:
        # 新发布（或被整体主动失效后重新发布）都进入 LRU 队尾。
        self._entries[key] = entry
        self._cost += entry.cost
        self.published += 1
        self._enforce_limits()

    # -------------------------------------------------- 对外操作
    def lookup(self, key: tuple[str, CompileOptions]) -> tuple[str, Any, Any]:
        now = self.config.time_func()
        with self._lock:
            self._purge_expired(now)
            value = self._entries.get(key)
            if isinstance(value, _Entry):
                self._entries.move_to_end(key)
                value.last_hit_at = now
                value.hits += 1
                self.hits += 1
                return ('hit', value.rule, None)
            if isinstance(value, _InFlight):
                self.misses += 1
                self.compile_waits += 1
                return ('wait', value, None)
            marker = _InFlight()
            self._entries[key] = marker
            self.misses += 1
            return ('compile', marker, None)

    def publish_success(
            self,
            key: tuple[str, CompileOptions],
            marker: _InFlight,
            rule: 'Rule',
            origin_context_id: int
    ) -> 'Rule':
        cost = self._cost_of(rule)
        now = self.config.time_func()
        expires_at = now + self.config.ttl if self.config.ttl > 0 else 0.0
        entry = _Entry(rule, cost, origin_context_id, now, expires_at)
        with self._lock:
            # 只有占位仍是自己的（或键在编译期间被主动失效清空）时才发布。
            current = self._entries.get(key)
            if current is marker or current is None:
                self._publish(key, entry)
            marker.rule = rule
            marker.event.set()
            stored = self._entries.get(key)
            return stored.rule if isinstance(stored, _Entry) else rule

    def publish_failure(self, key: tuple[str, CompileOptions], marker: _InFlight, exc: BaseException) -> None:
        with self._lock:
            self.compilation_failures += 1
            # 失败绝不留下任何已存储对象：只移除自己的占位，下一次请求重新编译。
            if self._entries.get(key) is marker:
                self._entries.pop(key, None)
            marker.exc = exc
            marker.event.set()

    def await_result(self, key: tuple[str, CompileOptions], marker: _InFlight) -> 'Rule':
        if not marker.event.wait(timeout=self.config.compile_wait_timeout):
            raise errors.EngineError('timed out waiting for an in-progress rule compilation')
        with self._lock:
            value = self._entries.get(key)
            if isinstance(value, _Entry):
                return value.rule
        # 结果可能在发布后立即被淘汰；协调者发布的对象仍可安全复用。
        if marker.rule is not None:
            return marker.rule
        assert marker.exc is not None
        raise marker.exc

    def invalidate(
            self,
            text: str | None = None,
            predicate: Callable[[str], bool] | None = None
    ) -> int:
        removed = 0
        with self._lock:
            for key, value in list(self._entries.items()):
                # 正在编译的占位不失效：其发布结果本身就是按最新输入计算的新对象。
                if not isinstance(value, _Entry):
                    continue
                if text is not None and key[0] != text:
                    continue
                if predicate is not None and not predicate(key[0]):
                    continue
                self._entries.pop(key)
                self._cost -= value.cost
                removed += 1
            self.invalidations += removed
        return removed

    def invalidate_context(self, context: 'Context') -> int:
        target = id(context)
        removed = 0
        with self._lock:
            for key, value in list(self._entries.items()):
                if isinstance(value, _Entry) and value.origin_context_id == target:
                    self._entries.pop(key)
                    self._cost -= value.cost
                    removed += 1
            self.invalidations += removed
        return removed

    def clear(self) -> int:
        with self._lock:
            removed = 0
            # 保留正在编译的占位：领导者完成后仍会正常发布其新结果。
            for key, value in list(self._entries.items()):
                if isinstance(value, _Entry):
                    self._entries.pop(key)
                    self._cost -= value.cost
                    removed += 1
            self.invalidations += removed
            return removed

    def prune(self) -> None:
        with self._lock:
            self._purge_expired(self.config.time_func())

    def _cost_of(self, rule: 'Rule') -> int:
        if self.config.cost_function is None:
            return self.config.default_cost
        weight = self.config.cost_function(rule)
        if weight < 0:
            raise ValueError('cost_function must return a non-negative integer')
        return int(weight)

    def snapshot(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                    current_entries=self._stored_count(),
                    current_cost=self._cost,
                    hits=self.hits,
                    misses=self.misses,
                    published=self.published,
                    compile_waits=self.compile_waits,
                    compilation_failures=self.compilation_failures,
                    evictions=self.evictions,
                    expirations=self.expirations,
                    invalidations=self.invalidations,
            )


class CompiledRuleCache(object):
    """进程级规则编译缓存管理器。

    :param config: 所有租户共享的容量/成本/过期策略；单个租户也可以在
        :py:meth:`get_or_compile` 时通过 *config* 覆盖（仅在该租户首次出现时
        生效）。
    """
    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._lock = threading.RLock()
        self._tenants: dict[str, _TenantCache] = {}
        self._closed = False

    def __len__(self) -> int:
        with self._lock:
            tenants = list(self._tenants.values())
        return sum(tenant._stored_count() for tenant in tenants)  # noqa: SLF001

    def __contains__(self, item: tuple[str, str]) -> bool:
        tenant_id, text = item
        with self._lock:
            tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        with tenant._lock:  # noqa: SLF001
            tenant._purge_expired(tenant.config.time_func())
            return any(key[0] == text and isinstance(value, _Entry) for key, value in tenant._entries.items())

    def _get_tenant(self, tenant_id: str, config: CacheConfig | None) -> _TenantCache:
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                if self._closed:
                    raise errors.EngineError('cache is closed')
                tenant = _TenantCache(tenant_id, config or self.config)
                self._tenants[tenant_id] = tenant
            return tenant

    # -------------------------------------------------- 编译入口
    def get_or_compile(
            self,
            tenant_id: str,
            text: str,
            context: 'Context | None' = None,
            *,
            parser: Any = None,
            config: CacheConfig | None = None
    ) -> 'Rule':
        """返回租户 *tenant_id* 下与 *text* / *context* 对应的已编译规则。

        未命中时以单飞方式编译：同一键上的并发调用只有一个线程真正编译，
        所有调用方拿到的是同一个可发布 :py:class:`~rule_engine.Rule`。
        编译在调用方上下文的隔离副本上进行，因此编译过程对
        :py:attr:`Context.symbols` 等可变状态的填充只体现在返回规则的
        :py:attr:`Rule.context` 上，不会回写调用方上下文，也不会与其它租户
        相互污染。编译失败时抛出原始异常且不产生任何缓存条目。
        """
        if self._closed:
            raise errors.EngineError('cache is closed')
        from .context import Context
        from .rule import Rule
        context = context if context is not None else Context()
        parser = parser or Rule.parser
        tenant = self._get_tenant(tenant_id, config)
        options = CompileOptions.from_context(context, parser)
        key = _CompositeKey(text, options)

        kind, payload, _ = tenant.lookup(key)
        if kind == 'hit':
            return payload
        if kind == 'wait':
            return tenant.await_result(key, payload)

        marker = payload
        # 编译在租户锁外执行：编译器只与隔离副本交互，缓存状态直到发布时才变更。
        compile_context = context.clone()
        try:
            rule = self._compile(text, compile_context, parser)
        except BaseException as exception:
            tenant.publish_failure(key, marker, exception)
            raise
        return tenant.publish_success(key, marker, rule, id(context))

    @staticmethod
    def _compile(text: str, context: 'Context', parser: Any) -> 'Rule':
        from .rule import Rule
        statement = parser.parse(text, context)
        return Rule(text, context, statement=statement)

    # -------------------------------------------------- 主动失效
    def invalidate(
            self,
            tenant_id: str,
            text: str | None = None,
            *,
            predicate: Callable[[str], bool] | None = None
    ) -> int:
        """主动失效单个租户的条目，返回被移除的条目数。

        *text* 为 ``None`` 且 *predicate* 为空时清空该租户全部已发布结果；
        否则可按文本精确失效或用 *predicate* 批量判定（调用方应已持有该租户
        的授权）。正在编译的键不受影响，其发布结果仍按最新输入计算。
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return 0
        return tenant.invalidate(text, predicate=predicate)

    def invalidate_context(self, context: 'Context', *, tenant_id: str | None = None) -> int:
        """失效使用给定 :py:class:`~rule_engine.Context` 实例编译的全部条目。"""
        with self._lock:
            tenants = list(self._tenants.values()) if tenant_id is None else [self._tenants.get(tenant_id)]
            tenants = [tenant for tenant in tenants if tenant is not None]
        return sum(tenant.invalidate_context(context) for tenant in tenants)

    def clear(self, tenant_id: str | None = None) -> int:
        """清空指定租户（缺省时清空全部租户）的已发布结果，返回移除数。"""
        with self._lock:
            tenants = list(self._tenants.values()) if tenant_id is None else [self._tenants.get(tenant_id)]
            tenants = [tenant for tenant in tenants if tenant is not None]
        return sum(tenant.clear() for tenant in tenants)

    # -------------------------------------------------- 维护与统计
    def prune(self, *, tenant_id: str | None = None) -> None:
        """主动清扫过期条目；正常使用中过期条目会在访问时惰性清除。"""
        with self._lock:
            tenants = list(self._tenants.values()) if tenant_id is None else [self._tenants.get(tenant_id)]
            tenants = [tenant for tenant in tenants if tenant is not None]
        for tenant in tenants:
            tenant.prune()

    def stats(self, *, tenant_id: str | None = None) -> CacheStats:
        """返回脱敏计数快照。

        指定 *tenant_id* 时返回该租户的计数；缺省返回进程级汇总，并在
        :py:attr:`CacheStats.per_tenant` 中给出每个租户的计数——其中只有数字，
        不包含任何租户的规则文本或上下文内容。
        """
        with self._lock:
            if tenant_id is not None:
                tenant = self._tenants.get(tenant_id)
                return dataclasses.replace(tenant.snapshot(), tenants=1) if tenant is not None else CacheStats()
            tenants = list(self._tenants.items())

        per_tenant: dict[str, CacheStats] = {}
        totals: dict[str, int] = collections.defaultdict(int)
        counted = tuple(
            field.name for field in dataclasses.fields(CacheStats) if field.name not in ('tenants', 'per_tenant')
        )
        for tid, tenant in tenants:
            snapshot = tenant.snapshot()
            per_tenant[tid] = snapshot
            for field_name in counted:
                totals[field_name] += getattr(snapshot, field_name)
        totals['tenants'] = len(tenants)
        return CacheStats(per_tenant=dict(per_tenant), **totals)

    def tenant_ids(self) -> tuple[str, ...]:
        """返回当前缓存中出现过的租户标识（不暴露任何规则信息）。"""
        with self._lock:
            return tuple(self._tenants)

    def close(self) -> None:
        """关闭缓存并释放全部条目；关闭后的编译请求将报错。"""
        with self._lock:
            self._closed = True
            for tenant in self._tenants.values():
                tenant.clear()
            self._tenants.clear()
