# Gate A official GPQA source inspection

2026-10-08。只授予executor01a11ac8-ca15-7573-8ff6-c1dc0892faab当前Gate A官方来源核验权限，取代先前补充中仅HF官方端点的限制，不改变正式科学绑定。

已收到证据：本地HF公开API200，候选revision83022cefff930aea54f654c0b282e74b9eeda5c6；对应原版CSV未认证HEAD401。远端HF连接reset无HTTP状态且既有token布尔false。上述事实不证明GitHub官方公开发布不可访问。

作者https://github.com/idavidrein/gpqa 的README明确公开提供dataset.zip与解压口令deserted-untie-orchid，并把HF列为替代入口。允许核对该作者仓库可读commit版本及文件大小后，以固定GitHub commit下载官方dataset.zip至Gate A既有远端新staging根，使用作者公开口令解压到独立新子目录；不下载到本地、不接受条款、不登录、不传凭证、不用第三方转载/代理。若包超过50MiB则先回报大小。只读检查Main成员文件、448行、字段和顺序约定，真实问题/答案不打印、不提交Git。标签字段仅限原A隔离输入构造边界。

该动作仅为source-inspection与候选绑定，不把GitHub commit冒充HF revision，不据题数相同就宣称内容完全等同。回报作者来源关系、实际commit/包内Main路径、数量/字段、可验证与不可验证的一致性边界；正式合同改绑仍由planning决定。若只有GitHub与HF内容相等无法核验，明确提出以作者原始发布为候选的新绑定供planning处理，不静默替换。其他A工作继续，以上无GPU/outcome权限。
