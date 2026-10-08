# Gate A GPQA access supplement

2026-10-08。仅适用于executor01a11ac8-ca15-7573-8ff6-c1dc0892faab，Gate A；不改变科学合同或开放GPU/outcome。

收到非终态访问报告：已查共享hub/datasets未见GPQA；既有token检查仅布尔false；metadata请求ConnectionError，无HTTP状态。当前仅为可见性/连接问题，不能据此判定HTTP访问拒绝。

允许同executor做最小有界排查，同时继续独立实现：

1. 按HPC2 skill只读检查huggingface.co官方端点DNS/TLS/HTTP，短timeout、有限次数，保留异常类型/状态，不输出凭证、Authorization header或完整环境。不得改DNS/代理/认证配置。
2. 只读搜索AGENTS允许的固定复现项目与LoopScope workspace中GPQA名称的manifest、输入路径、历史命令/任务配置，及shared hf_home/datasets目录索引；只读元数据与路径，不遍历输出原题、标签、答案、模型outcome或秘密文件。历史成果仅提供source定位，不作为Phase11评分结果复用。
3. 可在本地仅请求官方公开dataset metadata以区分远端网络问题；本地不下载数据/模型，不探测其他用户凭证、不代用户登录或接受访问条款。若发现合法既有输入，核对repo/revision/gpqa_main及读取权限，按原A授权在远端新input目录构建，不能用Diamond或任意镜像替代。
4. 普通连接故障允许短暂有限重试；没有现成合法数据且官方访问确需用户操作时，回报准确URL、状态、缺失项给planning。不得因该依赖阻塞停下其余可完成实现；全部独立工作完成后仍无路径，才发唯一Gate级BLOCK及已保留证据。

不扩大网络下载体量/环境安装/受保护代码权限；不需要额外审计平台或重复扫描。默认执行模型创建配置已改为gpt-6.1-sol/high，仅后续新聊天生效，当前A保持原配置。
