# 旅游联营咨询负荷的层级校准与借班承诺

联营接待按咨询链、服务时窗和语言席位复盘等待与借班。

## 资料和运行环境

`data/` 提供本地合成业务资料。各业务文件为 UTF-8 JSON 数组，键、单位和关联见 [资料字典](data/README.md)。`format.json` 记录列集合和基础引用约束。

使用 Python 3.11。在普通 Python 环境可用 `python3 -m pip install -r requirements.txt` 安装锁定的分析依赖。检查提供的资料可运行：

```sh
python3 -m unittest discover -s tests -v
```

资料检查覆盖文件可读性、字段与明确的单列引用。业务原始记录允许保留冲突、未决关系及尚未到达的观察。

## 分析

业务口径与结论见 [ANALYSIS.md](ANALYSIS.md)。重算全部指标并写出 `outputs/`：

```sh
python3 scripts/run_analysis.py --print
```

`tests/` 除了原有的资料契约检查，还有业务不变量（一个旅客一次时钟、等待分解可加、双席占用不重复计入、删失只给区间、借班与班表核对、切换不重启时限等），共 38 项。
