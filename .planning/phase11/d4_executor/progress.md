# Execution progress
2026-10-12: D4 binding confirmed, generation entrypoint committed/pushed. Targeted membership/command checks and bash syntax0. Debug12987865 COMPLETED0:0/24s;82tokens/81cached decode, no decode fitting, directions released.
Formal12987870/12987871 submitted, each1A800/8CPU/64G/2h;packing1;4GPUh totalreserved. RUNNING confirmed on gpu1-12/gpu1-41. Unique automation gate-d-4-gpqa-8192-full-monitor ACTIVE,60min, exact executor terminal-resume route.
See ../gate_d4_setup.json, ../gate_d4_job_ledger.json, ../gate_d4_monitor.json for recovery.

First A800 upper-tail jobs both0:0,788 allocation seconds=0.218889GPUh;7635/7069tokens. Peak10661/10484MiB of81920. New retained formal packing4 job12987981 (4new pairs) submitted. Pending full-membership split8058 after6 reserved first-batch pairs is staged; broad submission waits packed resource closure. Analysis5016be2 passed3CPU tests and is separate from immutable5fc00f0 runtime.

Packing5 formal job12988002 added,4new upper-tail pairs plus K3/current0.9 longest-input2819tokens/index344. Pending remaining8053 exact pairs and404 packing5 or504 packing4 batch candidates prepared remotely,4records per child; no outcome read. Immutable runtime unchanged.

Packing5 formal12988002 CLOSED0:0/987s,47100/81920MiB/headroom42.5percent, noOOM; maxchild7679tokens/981.94s and longestprompt2819 complete. Fullphase first8packing4 jobs12988334-12988341 (128pairs), then40packing5 jobs12988373-12988412 (800pairs) submitted successfully. With11resource records total939pairs submitted;7125remain. Candidate397batch p5remaining plan excludes all139preassigned pairs, nextbatch40. Runtime5fc00f0 unchanged; GPUh formalactual2594/3600=0.720556, currentreserved96GPUh beforelive settlement; no cumulative cap.

Observer refreshed to latest48jobs, failure/all-terminal/refill tail trigger, exact executor pause-and-wake confirmed protocol. Local Python lacks tomllib when inspecting automation; read existing TOML directly and used automation_update structured tool successfully, no automation file edited by shell.

Startup directly verified:48fulljobs,8RUNNING+40PENDING(QOSMaxCpuPerUserLimit), no failedparent;4fullchild records complete plus11resource=15verified records, fullpanel notclosed, goldunread. Currentplanfull-p5-remaining nextbatch40 and monitor ACTIVE.
