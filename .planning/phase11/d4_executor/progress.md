# Execution progress
2026-10-12: D4 binding confirmed, generation entrypoint committed/pushed. Targeted membership/command checks and bash syntax0. Debug12987865 COMPLETED0:0/24s;82tokens/81cached decode, no decode fitting, directions released.
Formal12987870/12987871 submitted, each1A800/8CPU/64G/2h;packing1;4GPUh totalreserved. RUNNING confirmed on gpu1-12/gpu1-41. Unique automation gate-d-4-gpqa-8192-full-monitor ACTIVE,60min, exact executor terminal-resume route.
See ../gate_d4_setup.json, ../gate_d4_job_ledger.json, ../gate_d4_monitor.json for recovery.

First A800 upper-tail jobs both0:0,788 allocation seconds=0.218889GPUh;7635/7069tokens. Peak10661/10484MiB of81920. New retained formal packing4 job12987981 (4new pairs) submitted. Pending full-membership split8058 after6 reserved first-batch pairs is staged; broad submission waits packed resource closure. Analysis5016be2 passed3CPU tests and is separate from immutable5fc00f0 runtime.

Packing5 formal job12988002 added,4new upper-tail pairs plus K3/current0.9 longest-input2819tokens/index344. Pending remaining8053 exact pairs and404 packing5 or504 packing4 batch candidates prepared remotely,4records per child; no outcome read. Immutable runtime unchanged.

Packing5 formal12988002 CLOSED0:0/987s,47100/81920MiB/headroom42.5percent, noOOM; maxchild7679tokens/981.94s and longestprompt2819 complete. Fullphase first8packing4 jobs12988334-12988341 (128pairs), then40packing5 jobs12988373-12988412 (800pairs) submitted successfully. With11resource records total939pairs submitted;7125remain. Candidate397batch p5remaining plan excludes all139preassigned pairs, nextbatch40. Runtime5fc00f0 unchanged; GPUh formalactual2594/3600=0.720556, currentreserved96GPUh beforelive settlement; no cumulative cap.

Observer refreshed to latest48jobs, failure/all-terminal/refill tail trigger, exact executor pause-and-wake confirmed protocol. Local Python lacks tomllib when inspecting automation; read existing TOML directly and used automation_update structured tool successfully, no automation file edited by shell.

Startup directly verified:48fulljobs,8RUNNING+40PENDING(QOSMaxCpuPerUserLimit), no failedparent;4fullchild records complete plus11resource=15verified records, fullpanel notclosed, goldunread. Currentplanfull-p5-remaining nextbatch40 and monitor ACTIVE.

2026-10-12 11:43UTC heartbeat observed all48initialfulljobs COMPLETED0:0; paused observer, confirmed AUTOMATION_TERMINAL_RESUME delivery to exactexecutor. Executor independently verified all928records/shards and all48parent closures;939formal records completed. Initialfull actual18.933333GPUh; resourceformal0.720556; debug0.006667; settledall19.660556GPUh. Runtime5fc00f0 remainsclean. Submitted48packing5 batches40-87, IDs12989391-12989438,960newpairs/96GPUhreservation. Total1899submitted,939complete,6165notyetsubmitted; nextbatch88.

Same60min observer reactivated after binding newwave40-87 IDs; snapshot1RUNNING+47PENDING immediately aftersubmit, nofailure,939complete. Added one read-only observer helper and exactwavejobset to prevent checking historicalreceipts afterfuture续批. Observer responseIDs exactly match localmonitor. Helper reads scheduler/summary only; no raw/gold/accuracy/MMLU.

2026-10-12 14:49UTC tail-refill wake:41wave40-87 jobs completed,7stillrunning; executor independently verified820wholejob records+28completed child records in carriedjobs, prior939 =>1787verified. Wholejob settledbaseline1759 used for nextobserver to avoid doublecounting carriedpartial children. Wave040 settled19.240278GPUh; accumulated settledall38.900833GPUh. Preserve7running; submit43packing5 batches88-130, jobs12990894-12990936 (860newpairs/86GPUhmaxreservation), total2759submitted,5305unsubmitted, next131. Newobserverjobset includes7carry+43new=50jobs; nofailedattempt/gold/MMLU.

Refill startup helper exactIDs verified:49inflight(6RUNNING+43PENDING), onecarriedjob nowterminal;1795child-record complete, nofailures/missingaccounting. Same60min observer ACTIVE on7carry+43new logicaljobset; no jobs cancelled/retried and no gold/MMLU.
