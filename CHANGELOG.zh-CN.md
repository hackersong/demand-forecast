# 更新日志

本文件记录项目的所有重要变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
English version: [CHANGELOG.md](CHANGELOG.md)

## [未发布]

### 新增

* 完整账号体系（`user_management`，issue #10）：自助注册（邮箱校验、
  Django 密码强度校验）、基于 auth 组的角色模型（admin / planner / viewer）
  并在所有写操作上强制校验、仅管理员可见的用户与角色管理页、
  用户仪表盘，以及修改密码 / 重置密码流程。
  附 29 项页面级测试，六门语言翻译同步更新。
* 预测引擎服务层（`model_training/services/`，issue #9）：
  `algorithms.py`（预测算法）、`metrics.py`（评估指标）、
  `trainer.py`（预处理与训练编排）、`constants.py`（层级映射与模型类型表）。
  `views.py` 从约 2500 行降至约 900 行并保留 re-export，
  `apps.demand_prediction` 的导入不受影响。
  拆分前后对纯函数（指标、Croston/SBA、趋势、季节分解、预处理、
  重试/回退参数表）做了输出快照对比，结果完全一致；
  另有 12 项不依赖 Django test client 的单元测试。
* Docker 支持（issue #7）：`Dockerfile`（python:3.12-slim + gettext +
  gunicorn）、带数据卷的 `docker-compose.yml`、启动时执行
  migrate + collectstatic 的 `docker-entrypoint.sh`、
  `DEBUG=False` 下由 WhiteNoise 提供静态文件，以及构建镜像并冒烟测试
  `/login/` 的 CI 任务。
* 多语言界面：默认英文，另有中文、Français、Deutsch、Español、日本語，
  可在运行时通过右上角下拉框切换。
* 六门语言的完整翻译目录，并提供 `scripts/sync_zh_catalog.py`
  用于同步简体中文目录。
* 本地化日期渲染（模板过滤器 `localize_date` / `localize_month`），
  月份与日期会跟随当前语言。
* `apps.core.context_processors.languages` 提供
  `AVAILABLE_LANGUAGES_NATIVE`，使切换器能以各语言自身名称显示。
* 中英文文档：README、安装部署、使用指南、架构说明与多语言说明。
* 开源社区文件：`LICENSE`（MIT）、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、
  `SECURITY.md`、Issue/PR 模板以及 GitHub Actions 工作流。
* `.env.example`，以及 `SECRET_KEY`、`DEBUG`、`ALLOWED_HOSTS`、`LANGUAGE_CODE`
  的环境变量化配置。
* 各模块界面截图（`docs/images/`），已在两份 README 与使用指南中引用
  （PR #11，感谢 @abhinavkumarpathak1-coder）。

### 变更

* 语言切换控件从侧边栏底部移到右上角，并改为高对比度（橙色）胶囊样式。
* 账号信息（用户名 + 退出登录）也从侧边栏底部移到右上角。
* 表格中的日期改为按当前语言格式化，不再是语言无关的 `%Y-%m` / `%Y-%m-%d`。
* 侧边栏菜单、页面标题与页脚文案也纳入翻译，中文界面不再残留英文导航项。
* 默认语言始终为 `LANGUAGE_CODE`（`en`）：除非开启
  `DJANGO_LANGUAGE_FROM_BROWSER`，否则忽略浏览器 `Accept-Language`
  （新增 `DefaultLanguageLocaleMiddleware`）。
* `SECRET_KEY` 不再硬编码，改为从 `DJANGO_SECRET_KEY` 读取
  （开发环境有不安全的兜底值）。
* `DEBUG` 与 `ALLOWED_HOSTS` 不再固定为 `True` / `['*']`。
* `Content-Language` 响应头跟随当前语言，不再固定为 `zh-CN`。
* `requirements.txt` 改为最低版本约束，并说明可选算法依赖。

### 修复

* `model_results` 与训练页抛出 `NameError`：服务层拆分时误删了
  `model_training/views.py` 中的 `from .models import ForecastModel, ForecastResult`。
  现已为两个页面补测试，并新增静态检查（`apps/core/tests.py`）：
  任一视图模块引用了任何作用域都未绑定的名字时测试直接失败。
* 界面文案被连续渲染 2～7 遍（首页提示、上传格式说明等）。原因是翻译目录的
  `msgstr` 同时写入了完整句子与自动换行的续行，gettext 拼接后同一句话重复多次。
  已修复全部六个目录（en 18、fr 27、de 21、es 26、ja 17 条），
  并重新生成简体中文目录。
* `scripts/sync_zh_catalog.py` 不再保留旧 `msgstr` 的续行，
  重新同步中文目录不会再引入重复。
* `upload_data` 视图中任何调用 `gettext` 的分支都会返回 HTTP 500：
  行循环把 gettext 别名 `_` 当作占位变量使用，使 `_` 变成局部变量，
  视图内所有 `_()` 调用都抛出 `UnboundLocalError`。
  PR #11（@abhinavkumarpathak1-coder）通过重命名循环变量修复。

### 安全

* `db.sqlite3`、`.env`、`__pycache__`、`media/`、`staticfiles/` 与编译产物
  `*.mo` 均已排除在版本控制之外。
