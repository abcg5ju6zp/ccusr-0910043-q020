#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  tests/cache.py
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

import datetime
import decimal
import functools
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import dateutil.tz

import rule_engine.engine as engine
import rule_engine.errors as errors
from rule_engine import CacheConfig, CompiledRuleCache, cache_key
from rule_engine.types import DataType

__all__ = ('CompiledRuleCacheTests',)

SECRET_TEXT = 'SECRET_TOKEN_4f92 == "value"'


class CompiledRuleCacheTests(unittest.TestCase):
    def _cache(self, **kwargs):
        return CompiledRuleCache(CacheConfig(**kwargs))

    # ------------------------------------------------------------ 基础命中
    def test_hit_returns_same_object(self):
        cache = self._cache()
        first = cache.get_or_compile('tenant-a', 'age > 18')
        second = cache.get_or_compile('tenant-a', 'age > 18')
        self.assertIs(first, second)
        stats = cache.stats(tenant_id='tenant-a')
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.published, 1)
        self.assertAlmostEqual(stats.hit_rate, 0.5)
        self.assertEqual(len(cache), 1)
        self.assertIn(('tenant-a', 'age > 18'), cache)

    def test_cached_rule_evaluates_correctly(self):
        cache = self._cache()
        rule = cache.get_or_compile('tenant-a', 'age >= 18 and name == "Luke"')
        self.assertTrue(rule.matches({'age': 20, 'name': 'Luke'}))
        self.assertFalse(rule.matches({'age': 12, 'name': 'Luke'}))

    # ------------------------------------------------------------ 租户隔离
    def test_tenants_are_isolated(self):
        cache = self._cache()
        rule_a = cache.get_or_compile('tenant-a', 'age > 18')
        rule_b = cache.get_or_compile('tenant-b', 'age > 18')
        self.assertIsNot(rule_a, rule_b)
        self.assertIsNot(rule_a.context, rule_b.context)
        # 两个租户各自发生一次 miss，而不是后者命中前者
        self.assertEqual(cache.stats(tenant_id='tenant-a').misses, 1)
        self.assertEqual(cache.stats(tenant_id='tenant-b').misses, 1)
        self.assertEqual(cache.stats(tenant_id='tenant-a').hits, 0)

    def test_one_tenant_flooding_does_not_evict_another(self):
        cache = self._cache(max_entries=4)
        pinned = cache.get_or_compile('tenant-a', 'pinned == 1')
        for i in range(100):
            cache.get_or_compile('tenant-b', 'flood%d == 1' % i)
        # 租户 A 的条目没有被租户 B 的批量唯一规则挤掉
        self.assertIn(('tenant-a', 'pinned == 1'), cache)
        self.assertIs(cache.get_or_compile('tenant-a', 'pinned == 1'), pinned)
        stats_a = cache.stats(tenant_id='tenant-a')
        self.assertEqual(stats_a.evictions, 0)
        self.assertEqual(stats_a.hits, 1)

    def test_invalidate_and_clear_are_scoped_to_tenant(self):
        cache = self._cache()
        rule_a = cache.get_or_compile('tenant-a', 'x == 1')
        cache.get_or_compile('tenant-b', 'x == 1')
        self.assertEqual(cache.invalidate('tenant-a', 'x == 1'), 1)
        self.assertNotIn(('tenant-a', 'x == 1'), cache)
        self.assertIn(('tenant-b', 'x == 1'), cache)
        # 失效后重新编译产生新对象
        self.assertIsNot(cache.get_or_compile('tenant-a', 'x == 1'), rule_a)
        cache.clear('tenant-a')
        self.assertEqual(len(cache), 1)
        cache.clear()
        self.assertEqual(len(cache), 0)

    # ------------------------------------------------------------ 编译上下文隔离
    def test_equivalent_default_contexts_share(self):
        cache = self._cache()
        first = cache.get_or_compile('t', 'x == 1', engine.Context())
        second = cache.get_or_compile('t', 'x == 1', engine.Context())
        self.assertIs(first, second)

    def test_regex_flags_participate_in_key(self):
        import re
        cache = self._cache()
        plain = cache.get_or_compile('t', 'name =~ "abc"', engine.Context(regex_flags=0))
        ci = cache.get_or_compile('t', 'name =~ "abc"', engine.Context(regex_flags=re.IGNORECASE))
        self.assertIsNot(plain, ci)
        self.assertFalse(plain.matches({'name': 'ABC'}))
        self.assertTrue(ci.matches({'name': 'ABC'}))
        # 相同 flags 仍然共享
        self.assertIs(
                cache.get_or_compile('t', 'name =~ "abc"', engine.Context(regex_flags=re.IGNORECASE)),
                ci
        )

    def test_mapping_attribute_lookup_participates_in_key(self):
        cache = self._cache()
        enabled = cache.get_or_compile('t', 'f.a == 1', engine.Context(mapping_attribute_lookup=True))
        disabled = cache.get_or_compile('t', 'f.a == 1', engine.Context(mapping_attribute_lookup=False))
        self.assertIsNot(enabled, disabled)

    def test_timezone_participates_in_key(self):
        cache = self._cache()
        utc = engine.Context(default_timezone=dateutil.tz.tzutc())
        offset = engine.Context(default_timezone=dateutil.tz.tzoffset(None, 3600))
        text = 'd("2024-01-01T00:30:00") > d("2024-01-01T00:00:00")'
        self.assertIsNot(cache.get_or_compile('t', text, utc), cache.get_or_compile('t', text, offset))
        # 等价的 tzutc 实例仍然共享
        self.assertIs(
                cache.get_or_compile('t2', text, engine.Context(default_timezone=dateutil.tz.tzutc())),
                cache.get_or_compile('t2', text, engine.Context(default_timezone=dateutil.tz.tzutc()))
        )

    def test_default_value_participates_in_key(self):
        cache = self._cache()
        missing_default = engine.Context(default_value=None)
        zero_default = engine.Context(default_value=0)
        rule_none = cache.get_or_compile('t', 'missing == null', missing_default)
        rule_zero = cache.get_or_compile('t', 'missing == null', zero_default)
        self.assertIsNot(rule_none, rule_zero)
        self.assertTrue(rule_none.matches({}))
        self.assertFalse(rule_zero.matches({}))

    def test_decimal_context_participates_in_key(self):
        cache = self._cache()
        default_ctx = engine.Context()
        low_prec = engine.Context(decimal_context=decimal.Context(prec=2))
        rule = cache.get_or_compile('t', 'x == 1', default_ctx)
        self.assertIsNot(rule, cache.get_or_compile('t', 'x == 1', low_prec))

    def test_resolver_participates_in_key_by_identity_and_registration(self):
        cache = self._cache()
        default_rule = cache.get_or_compile('t', 'name == "x"', engine.Context())
        attr_rule = cache.get_or_compile('t', 'name == "x"', engine.Context(resolver=engine.resolve_attribute))
        self.assertIsNot(default_rule, attr_rule)

        @cache_key('resolver-v1')
        def resolver_v1(thing, name):
            return thing[name]

        @cache_key('resolver-v2')
        def resolver_v2(thing, name):
            return thing[name]

        # 相同注册身份的两个函数被视为同一编译上下文
        @cache_key('resolver-v1')
        def resolver_v1_again(thing, name):
            return thing[name]

        c1 = engine.Context(resolver=resolver_v1)
        c2 = engine.Context(resolver=resolver_v1_again)
        c3 = engine.Context(resolver=resolver_v2)
        self.assertIs(cache.get_or_compile('t', 'name == "x"', c1),
                      cache.get_or_compile('t', 'name == "x"', c2))
        self.assertIsNot(cache.get_or_compile('t', 'name == "x"', c1),
                         cache.get_or_compile('t', 'name == "x"', c3))

    def test_type_resolver_mapping_participates_in_key(self):
        cache = self._cache()
        float_types = engine.Context(type_resolver={'x': DataType.FLOAT})
        str_types = engine.Context(type_resolver={'x': DataType.STRING})
        same_types = engine.Context(type_resolver={'x': DataType.FLOAT})
        first = cache.get_or_compile('t', 'x', float_types)
        # 不同类型映射产生不同的编译结果
        other = cache.get_or_compile('t', 'x', str_types)
        self.assertIsNot(first, other)
        self.assertEqual(first.statement.expression.result_type, DataType.FLOAT)
        self.assertEqual(other.statement.expression.result_type, DataType.STRING)
        # 内容相同的映射仍然共享
        self.assertIs(first, cache.get_or_compile('t', 'x', same_types))

    def test_compile_uses_isolated_context_copy(self):
        cache = self._cache()
        caller_context = engine.Context(type_resolver={'x': DataType.FLOAT})
        rule = cache.get_or_compile('t', 'x > 0', caller_context)
        # 编译期的符号收集只落在隔离副本上，不回写调用方上下文
        self.assertEqual(caller_context.symbols, set())
        self.assertEqual(rule.context.symbols, {'x'})
        self.assertIsNot(rule.context, caller_context)
        # 隔离副本的内置取值器仍然可用（regex 分组走线程本地存储）
        regex_rule = cache.get_or_compile('t2', r'words =~ "(\w+)" and $re_groups[0] == "Main"')
        self.assertTrue(regex_rule.matches({'words': 'Main'}))
        self.assertFalse(regex_rule.matches({'words': 'Other'}))

    # ------------------------------------------------------------ 容量与成本
    def test_max_entries_lru_eviction(self):
        cache = self._cache(max_entries=2)
        a = cache.get_or_compile('t', 'a == 1')
        cache.get_or_compile('t', 'b == 2')
        cache.get_or_compile('t', 'a == 1')  # 命中，a 变为最近使用
        cache.get_or_compile('t', 'c == 3')  # 应淘汰最久未使用的 b
        self.assertIn(('t', 'a == 1'), cache)
        self.assertNotIn(('t', 'b == 2'), cache)
        self.assertIn(('t', 'c == 3'), cache)
        self.assertEqual(cache.stats(tenant_id='t').evictions, 1)
        # 被淘汰对象仍可安全使用（淘汰只移除缓存字典中的引用）
        self.assertTrue(a.matches({'a': 1}))
        # 直接持有的另一条被淘汰规则同样可用
        b = self._evicted_helper()
        self.assertTrue(b.matches({'b': 2}))

    def _evicted_helper(self):
        cache = self._cache(max_entries=1)
        b = cache.get_or_compile('t', 'b == 2')
        cache.get_or_compile('t', 'a == 1')
        return b

    def test_max_cost_weight_eviction(self):
        cache = self._cache(max_entries=0, max_cost=10, cost_function=lambda rule: len(rule.text))
        cache.get_or_compile('t', 'aaaaaa')  # 6
        cache.get_or_compile('t', 'bbbb')    # 4（累计 10）
        cache.get_or_compile('t', 'cc')      # 2（超出，淘汰最旧的 aaaaaa）
        stats = cache.stats(tenant_id='t')
        self.assertNotIn(('t', 'aaaaaa'), cache)
        self.assertIn(('t', 'bbbb'), cache)
        self.assertIn(('t', 'cc'), cache)
        self.assertEqual(stats.current_cost, 6)
        self.assertEqual(stats.evictions, 1)

    def test_negative_cost_is_rejected(self):
        cache = self._cache(cost_function=lambda rule: -1)
        with self.assertRaises(ValueError):
            cache.get_or_compile('t', 'x == 1')

    # ------------------------------------------------------------ 过期
    def test_ttl_expiration(self):
        clock = [1000.0]
        cache = self._cache(ttl=10.0, time_func=lambda: clock[0])
        first = cache.get_or_compile('t', 'x == 1')
        clock[0] += 9
        self.assertIs(cache.get_or_compile('t', 'x == 1'), first)
        clock[0] += 2  # 超过 TTL
        second = cache.get_or_compile('t', 'x == 1')
        self.assertIsNot(first, second)
        self.assertEqual(cache.stats(tenant_id='t').expirations, 1)

    def test_prune_removes_expired_entries(self):
        clock = [0.0]
        cache = self._cache(ttl=5.0, time_func=lambda: clock[0])
        cache.get_or_compile('t', 'x == 1')
        clock[0] = 6
        cache.prune(tenant_id='t')
        self.assertNotIn(('t', 'x == 1'), cache)
        self.assertEqual(cache.stats(tenant_id='t').expirations, 1)

    # ------------------------------------------------------------ 主动失效
    def test_invalidate_with_predicate(self):
        cache = self._cache()
        cache.get_or_compile('t', 'alpha == 1')
        cache.get_or_compile('t', 'beta == 2')
        cache.get_or_compile('t', 'alpha2 == 3')
        removed = cache.invalidate('t', predicate=lambda text: text.startswith('alpha'))
        self.assertEqual(removed, 2)
        self.assertNotIn(('t', 'alpha == 1'), cache)
        self.assertNotIn(('t', 'alpha2 == 3'), cache)
        self.assertIn(('t', 'beta == 2'), cache)

    def test_invalidate_context(self):
        cache = self._cache()
        context = engine.Context(type_resolver={'x': DataType.FLOAT})
        other_context = engine.Context()
        rule = cache.get_or_compile('t', 'x', context)
        other = cache.get_or_compile('t2', 'y', other_context)
        self.assertEqual(cache.invalidate_context(context), 1)
        self.assertNotIn(('t', 'x'), cache)
        # 其它上下文（以及其它租户）的条目不受影响
        self.assertIn(('t2', 'y'), cache)
        self.assertIs(cache.get_or_compile('t2', 'y', other_context), other)
        # 失效后使用同一上下文重新编译产生新对象，并重新登记来源
        new_rule = cache.get_or_compile('t', 'x', context)
        self.assertIsNot(new_rule, rule)
        self.assertEqual(cache.invalidate_context(context), 1)

    def test_invalidate_does_not_duplicate_inflight_compile(self):
        gate = threading.Event()
        entered = threading.Event()

        @cache_key('inflight-types')
        def gated_types(name):
            entered.set()
            gate.wait(5.0)
            return DataType.FLOAT

        cache = self._cache()
        context = engine.Context(type_resolver=gated_types)
        leader = threading.Thread(target=lambda: cache.get_or_compile('t', 'x', context))
        leader.start()
        self.assertTrue(entered.wait(5.0))
        # 编译进行中执行整体失效，不得移除占位或产生重复编译窗口
        self.assertEqual(cache.invalidate('t'), 0)
        waiter = threading.Thread(target=lambda: cache.get_or_compile('t', 'x', context))
        waiter.start()
        gate.set()
        leader.join(5.0)
        waiter.join(5.0)
        self.assertFalse(leader.is_alive() or waiter.is_alive())
        stats = cache.stats(tenant_id='t')
        # 只有领导者真正编译了一次（publish == 1），等待者命中同一结果
        self.assertEqual(stats.published, 1)

    # ------------------------------------------------------------ 单飞
    def test_concurrent_compile_is_single_flight(self):
        gate = threading.Event()
        calls = []

        @cache_key('gated-resolver')
        def gated_resolver(name):
            calls.append(name)
            gate.wait(5.0)
            return DataType.UNDEFINED

        context = engine.Context(type_resolver=gated_resolver)
        cache = self._cache()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(cache.get_or_compile, 't', 'foo > 1', context) for _ in range(8)]
            # 等待领导者进入编译
            for _ in range(50):
                if calls:
                    break
                threading.Event().wait(0.02)
            self.assertEqual(len(calls), 1)
            gate.set()
            rules = [future.result() for future in futures]
        first = rules[0]
        self.assertTrue(all(rule is first for rule in rules))
        stats = cache.stats(tenant_id='t')
        self.assertEqual(stats.published, 1)
        self.assertEqual(stats.misses, 8)
        self.assertEqual(stats.compile_waits, 7)
        self.assertEqual(len(calls), 1)

    def test_waiters_receive_leader_failure_then_retry_succeeds(self):
        gate = threading.Event()
        state = {'entries': 0, 'fail': True}

        @cache_key('flaky-resolver')
        def flaky(name):
            state['entries'] += 1
            if state['fail']:
                gate.wait(5.0)
                raise errors.SymbolResolutionError(name)
            return DataType.UNDEFINED

        context = engine.Context(type_resolver=flaky)
        cache = self._cache()

        def attempt():
            return cache.get_or_compile('t', 'foo > 1', context)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(attempt) for _ in range(4)]
            for _ in range(50):
                if state['entries'] >= 1:
                    break
                threading.Event().wait(0.02)
            gate.set()
            for future in futures:
                with self.assertRaises(errors.SymbolResolutionError):
                    future.result()

        # 失败绝不污染缓存：下一次请求重新尝试，且最终可以成功
        self.assertNotIn(('t', 'foo > 1'), cache)
        stats = cache.stats(tenant_id='t')
        self.assertEqual(stats.published, 0)
        self.assertGreaterEqual(stats.compilation_failures, 1)
        state['fail'] = False
        rule = attempt()
        self.assertTrue(rule.matches({'foo': 2}))
        self.assertIn(('t', 'foo > 1'), cache)

    # ------------------------------------------------------------ 淘汰与评估并发
    def test_eviction_concurrent_with_evaluation_is_safe(self):
        cache = self._cache(max_entries=5)
        held = [cache.get_or_compile('t', 'n%d == 1' % i) for i in range(5)]
        problems = []

        def evaluate_held():
            for _ in range(200):
                for i, rule in enumerate(held):
                    try:
                        if not rule.matches({'n%d' % i: 1}):
                            problems.append(('wrong-true', i))
                        if rule.matches({'n%d' % i: 0}):
                            problems.append(('wrong-false', i))
                    except Exception as exc:  # noqa: BLE001 - 任何异常都视为失败
                        problems.append((type(exc).__name__, str(exc)))

        def flood_compiles():
            for j in range(300):
                rule = cache.get_or_compile('t', 'm%d == 1' % j)
                if not rule.matches({'m%d' % j: 1}):
                    problems.append(('flood-wrong', j))

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda fn: fn(), [evaluate_held, flood_compiles, evaluate_held, flood_compiles]))
        self.assertEqual(problems, [])
        # 被持续持有的 5 条规则在缓存之外仍然全部可用
        for i, rule in enumerate(held):
            self.assertTrue(rule.matches({'n%d' % i: 1}))

    # ------------------------------------------------------------ 统计脱敏
    def test_stats_do_not_leak_rule_text(self):
        cache = self._cache()
        cache.get_or_compile('tenant-a', SECRET_TEXT)
        cache.get_or_compile('tenant-a', 'other == 1')
        cache.get_or_compile('tenant-b', SECRET_TEXT)
        try:
            cache.get_or_compile('tenant-a', '((( invalid')
        except errors.EngineError:
            pass
        aggregate = cache.stats()
        self.assertEqual(aggregate.tenants, 2)
        self.assertEqual(set(aggregate.per_tenant), {'tenant-a', 'tenant-b'})
        blobs = [repr(aggregate), repr(aggregate.per_tenant)]
        blobs.extend(repr(snapshot) for snapshot in aggregate.per_tenant.values())
        for blob in blobs:
            self.assertNotIn('SECRET_TOKEN_4f92', blob)
            self.assertNotIn('other == 1', blob)
        # 单租户视图也不含文本
        self.assertNotIn('SECRET_TOKEN_4f92', repr(cache.stats(tenant_id='tenant-a')))
        self.assertNotIn('SECRET_TOKEN_4f92', repr(cache.tenant_ids()))

    def test_aggregate_stats_counts(self):
        cache = self._cache(max_entries=2)
        cache.get_or_compile('a', 'x == 1')
        cache.get_or_compile('a', 'x == 1')
        cache.get_or_compile('b', 'y == 2')
        cache.get_or_compile('b', 'z == 3')
        cache.get_or_compile('b', 'w == 4')  # 触发一次淘汰
        stats = cache.stats()
        self.assertEqual(stats.tenants, 2)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 4)
        self.assertEqual(stats.evictions, 1)
        self.assertEqual(stats.current_entries, len(cache))

    # ------------------------------------------------------------ 关闭
    def test_close_releases_entries_and_rejects_compiles(self):
        cache = self._cache()
        cache.get_or_compile('t', 'x == 1')
        cache.close()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.stats().tenants, 0)
        with self.assertRaises(errors.EngineError):
            cache.get_or_compile('t', 'x == 1')


if __name__ == '__main__':
    unittest.main()
