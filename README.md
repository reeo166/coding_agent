Git仓库：https://github.com/reeo166/coding_agent

这是一个从零实现的本地编程智能体，不依赖 agent 框架或云端代码执行服务。它通过 DeepSeek Chat Completions 原生 tool calling 完成多轮“模型-工具-模型”循环，并正确回传思考模式的 reasoning_content。内置读文件、列目录、文本搜索、写文件、精确替换和本地命令工具，具备上下文压缩、workspace 路径隔离、敏感文件屏蔽、超时、重试及循环上限。命令固定从 workspace 启动且不继承密钥，但它不是 OS 沙箱；只对可信任务使用 --yes。

运行环境：Python 3.10+，无第三方运行依赖。

API 接入：登录 https://platform.deepseek.com/api_keys 创建密钥并确认账户可用余额。复制 .env.example 为 .env，填写 DEEPSEEK_API_KEY；默认地址为 https://api.deepseek.com，默认模型为 deepseek-v4-flash，也可改成 deepseek-v4-pro。旧的 CODING_AGENT_API_KEY/BASE_URL/MODEL 变量仍兼容.

检查配置：python main.py --config .env --check-config
单次任务：python main.py --config .env --yes "检查项目并修复测试"
交互模式：python main.py --config .env
指定目录：python main.py --config 配置路径 --workspace 项目路径 "任务"
运行测试：python -m unittest discover -v
交互命令：:clear 清空上下文，:quit 退出。若接口返回 401/404/429，程序会提示检查 Key、地址、模型、权限或额度。
