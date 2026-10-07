# 规则表达式引擎

本项目提供可嵌入服务端的规则解析、类型检查、属性解析和表达式评估能力。生产源码位于 `lib/rule_engine/`，核心回归测试位于 `tests/`。

## 安装

`python3 -m pip install --break-system-packages --no-build-isolation -e .`

## 测试

`python3 -m pytest -q`

## 构建

`python3 -m compileall -q lib/rule_engine`

`python3 -m build --wheel --no-isolation`

## 使用

调用方创建规则并传入普通 Python 对象即可完成本地评估，不需要外部服务。

```python
import rule_engine

rule = rule_engine.Rule('user.age >= 18 and user.country == "CN"')
rule.matches({'user': {'age': 20, 'country': 'CN'}})  # True
```

## 多租户编译缓存（规则网关）

`CompiledRuleCache` 为规则网关提供按**租户**与**编译上下文**双重隔离的编译缓存，
支持容量上限、成本权重、TTL 过期与主动失效：

```python
from rule_engine import CompiledRuleCache, CacheConfig

cache = CompiledRuleCache(CacheConfig(max_entries=512, max_cost=4096, ttl=300))
rule = cache.get_or_compile(tenant_id='tenant-a', text='user.age >= 18', context=context)
rule.matches({'user': {'age': 18}})

cache.invalidate('tenant-a', 'user.age >= 18')   # 精确失效
cache.invalidate('tenant-a', predicate=lambda t: t.startswith('user.'))  # 批量失效
cache.clear('tenant-a')                          # 清空单个租户
cache.stats()                                    # 脱敏的进程级计数（不含规则文本）
```

关键语义：

- 缓存键为 `(租户, 规则文本, CompileOptions)`；`CompileOptions` 覆盖 `Context`
  中所有会改变编译结果的选项（regex_flags、时区、default_value、decimal 上下文、
  resolver、type_resolver、内置表等），不同解析上下文生成的对象永不串用。
  闭包形式的 resolver / type_resolver 可用 `@cache_key('稳定身份')` 声明等价关系。
- 同一键的并发编译只有一个线程真正执行（single-flight），等待者共享同一结果。
- 编译失败只回滚占位、不留下任何缓存对象，后续请求会重新尝试。
- 被淘汰或失效的只移除缓存内引用；已经取出的 `Rule` 在并发评估期间始终安全可用。

