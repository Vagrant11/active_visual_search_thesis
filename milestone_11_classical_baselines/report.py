"""逐φ等权汇总和不挑选样本的图。"""
import json
import numpy as np
from experiment import HERE, CAP, csv_write, json_write

ORDER = ['lawn_fast', 'lawn_conservative', 'greedy_probability', 'greedy_probability_per_distance',
         'ETO', 'diffusion_guided_smooth_scale', 'nearest', 'diffusion_guided_raw']
LABEL = dict(lawn_fast='割草机：快速', lawn_conservative='割草机：保守',
    greedy_probability='贪心：概率', greedy_probability_per_distance='贪心：概率/路程',
    ETO='ETO（5条）', diffusion_guided_smooth_scale='Diffusion引导＋平滑缩放（8条）',
    nearest='最近邻', diffusion_guided_raw='附：Diffusion仅引导（8条）')
ENGLISH = dict(lawn_fast='Lawn fast', lawn_conservative='Lawn conservative',
    greedy_probability='Greedy probability', greedy_probability_per_distance='Greedy probability/distance',
    ETO='ETO', diffusion_guided_smooth_scale='Guided diffusion + smooth/scale',
    nearest='Nearest', diffusion_guided_raw='Guided diffusion raw')
METRICS = ['detection_probability_common', 'conditional_mean_t_find_common', 'detection_probability_120',
           'reach_time', 'restricted_reach_time', 'generation_seconds', 'max_speed', 'max_abs_acceleration_component']


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def fmt(v, digits=3):
    return 'NA' if v is None else f'{v:.{digits}f}'


def report(data, rows, references, archived, env):
    per_phi = []
    summary = []
    for method in ORDER:
        selected = [r for r in rows if r['method'] == method]
        for scope in ('all', 'physical_only'):
            group = selected if scope == 'all' else [r for r in selected if r['physical_pass']]
            phi_rows = []
            for seed in sorted(set(r['phi_seed'] for r in selected)):
                subset = [r for r in group if r['phi_seed'] == seed]
                if not subset: continue
                entry = dict(method=method, scope=scope, phi_seed=seed, n=len(subset),
                    physical_pass_rate=mean([r['physical_pass'] for r in subset]),
                    reached_rate=mean([r['reached'] for r in subset]))
                for key in METRICS: entry[key] = mean([r[key] for r in subset])
                entry['reach_time_defined_samples'] = sum(r['reach_time'] is not None for r in subset)
                phi_rows.append(entry)
                per_phi.append(entry)
            s = dict(method=method, scope=scope, n=len(group), phi_count=len(phi_rows),
                physical_pass_rate=mean([r['physical_pass'] for r in group]),
                reached_samples=sum(r['reached'] for r in group),
                reached_rate=mean([r['reached_rate'] for r in phi_rows]),
                reach_time_defined_phi=sum(r['reach_time'] is not None for r in phi_rows))
            for key in METRICS:
                values = [r[key] for r in phi_rows if r[key] is not None]
                for label, function in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max), ('median', np.median)]:
                    s[key+'_'+label] = float(function(values)) if values else None
            summary.append(s)
    csv_write(HERE/'per_phi.csv', per_phi)
    csv_write(HERE/'summary.csv', summary)
    json_write(HERE/'summary.json', summary)
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    primitive = json.loads((HERE/'primitive_checks.json').read_text())
    lines = ['# M11 传统搜索基线：M8 val 共同预算对照', '',
        '仅使用 M8 的20个val φ；旧ETO、最近邻、diffusion均读取已存轨迹，不求解、不训练、不使用test。'
        '四个传统设置各评测20个φ；两版割草机各生成一条共享轨迹，两种贪心各生成20条，共42条新增轨迹。', '',
        '## 冻结规则与复用', '',
        '- 相机R=0.25、FoV=90°、dt=0.05s、独立扫描π/2 rad/s；真实矩形[.43,.57]²，安全圆中心(.5,.5)、半径0.11399494936611662。读取M9环境记录，并由M3 Environment构造安全圆。',
        '- 速度范数≤1，加速度每分量绝对值≤1；上限从实际planner约束AST读取。M3 planner.py:60–67；external/time_optimal_ergodic_search/experiments/comparison_study/build_solver.py:88–118。',
        '- 原M4 first_seen_grid和weighted_detection_stats按原AST执行；在线后验更新直接使用first_seen_grid的单次观测循环体。原M5 geometry_audit检查完整线段；原M7 geometry_metrics检查重建速度和控制。',
        '- 主表不施加E≤.05门槛；任务是搜索检测比较。物理通过包括边界、起点/初速度、速度/加速度、矩形避障和安全圆约束，不包含固定终点或ergodic门槛。', '',
        '## 四个传统策略', '',
        '割草机两版共享蛇形路点：y=.125,.375,.625,.875，左右行端x=0,1，自下而上。快速版每直线段采用最短离散时间的三角/梯形速度包络，不因相机旋转限速，行端仅降至零速转向，无扫描等待。'
        '保守版速度上限0.9×2√(R²−(.25/2)²)/4≈.09743，每个行端停留一整圈4s。两版一轮后都按避障路程最近的未见自由格子补扫，到达后等待该格子被看到；这一步完全不读取φ。快速版取消的是行端额外等待，补扫的必要观测等待仍保留。', '',
        '贪心概率版取未见格子的后验概率最大值；路程版取后验概率/避障可见图最短路程最大值。路程小于1e-12时用1e-12作分母，同分按网格编号。'
        '每0.05s用原可见性更新，已观测未发现格子清零；每次决策对剩余质量归一化。目标持续到被观测或到达，没有0.125步长停车规则。'
        '目标提前可见时立即重选；若新方向不同，沿当前安全直线制动后重选并转向；若方向相同且留有制动距离，则保持非零速度继续。到达但相机尚未扫到所选格子时原地等待，最多一圈。'
        '没有真实目标位置输入，评测用原始φ加权整条“未发现”分支的首次可见时间。', '',
        '避障路由使用64个安全圆外切多边形顶点作为候选，经整段圆净空检查构成可见图，额外径向余量1e-4。最短路程指该有限可见图上的最短路径，是连续安全圆最短路程的离散近似。'
        '折线拐角降至零速，不能将路径突然折向解释为有限加速度。每段取方向d，对应标量加速度上限a=1/max(|d_x|,|d_y|)，速度上限1；搜索最少整数步的可行包络，精确匹配线段长度。'
        '位置和速度遵守Euler递推；本结论针对planner的离散动力学，未声称折线插值在数学上处处具有连续加速度。', '',
        '所有传统轨迹初速度和末速度为零，但终点位置自由；ETO、NN及diffusion沿用固定(.9,.9)终点。这是策略约束不对称，结果不能解释成相同终点约束下的纯算法优劣。'
        '达到120s时仅在保存文件中追加安全制动尾段，不把尾段观测计入120s指标。格子观测停靠点统一向+x偏移1e-6，远小于0.025网格间距，避免目标与相机严格共点时方位未定义、被浮点插值左右。'
        '仍由原可见性判断发现，不将到达视为发现；这一数值规则对两种贪心和两种割草机补扫相同，未用val调参。', '',
        '## 共同预算与目标', '',
        '对每个φ，Tφ取五条ETO的tf算术平均；目标Pφ取这五条ETO在同一Tφ的检测概率平均。所有轨迹保留自身时序，超过Tφ截断，提前结束则末端停留并持续扫描；没有把tf重缩放到Tφ。'
        '条件Tfind为已检测质量的加权平均，始终与Pdet一起报告。每条轨迹首次达到Pφ的观测时间另存；120s未达到记为空和reached=false，不把失败误记成120s成功。', '',
        f'Tφ范围：{min(r["common_time"] for r in references):.4f}–{max(r["common_time"] for r in references):.4f}s；均值{np.mean([r["common_time"] for r in references]):.4f}s。'
        f'目标Pφ均值{np.mean([r["target_probability"] for r in references]):.5f}。', '',
        '汇总先在每φ内平均，再在φ间等权平均。达到时间仅对达到者定义，表中同时给达到率；另列min(达到时间,120)的限制平均，未达到者按120计入，以避免仅看成功样本的偏差。'
        '物理通过子集单独汇总，不能将不安全轨迹的检测收益当作可执行性能。', '',
        '## 主表：全部样本，失败保留', '',
        '|方法|物理通过|Pdet(Tφ)|条件Tfind(s)|120s达到目标|达到者时间(s)|限制平均(s)|计算耗时(s)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in ORDER:
        r = next(r for r in summary if r['method'] == method and r['scope'] == 'all')
        lines.append(f'|{LABEL[method]}|{r["physical_pass_rate"]:.1%}|{fmt(r["detection_probability_common_mean"],4)}|'
            f'{fmt(r["conditional_mean_t_find_common_mean"])}|{r["reached_rate"]:.1%}|{fmt(r["reach_time_mean"])}|'
            f'{fmt(r["restricted_reach_time_mean"])}|{fmt(r["generation_seconds_mean"],6)}|')
    lines += ['', 'Diffusion主行使用guidance既有val选择η=.5、无末次投影，再接既有一次平滑＋时间缩放，保留每φ全部8条；附行保留仅引导轨迹。'
        '选择来自旧val实验，没有本次重新选强度或挑样本；这20个val不是独立泛化验证集。', '',
        'ETO的目标是同φ五条解的平均检测概率，不是每条解自己的检测概率；因此某些低于平均值的ETO路径，即使末端继续扫描，也达不到这一目标。'
        'ETO达到率70%不表示求解失败，100条ETO物理检查全部通过。达到时间采用共同的0.05s观测网格，Tφ处另按原harness加入精确截止观测。', '',
        '## 仅物理通过的样本（描述性子集）', '',
        '|方法|保留轨迹/原分母|有可行样本φ数|Pdet(Tφ)|条件Tfind(s)|达到率|达到者时间(s)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in ORDER:
        r = next(r for r in summary if r['method'] == method and r['scope'] == 'physical_only')
        total = sum(x['method'] == method for x in rows)
        reach = 'NA' if r['reached_rate'] is None else f'{100*r["reached_rate"]:.1f}%'
        lines.append(f'|{LABEL[method]}|{r["n"]}/{total}|{r["phi_count"]}/20|{fmt(r["detection_probability_common_mean"],4)}|'
            f'{fmt(r["conditional_mean_t_find_common_mean"])}|{reach}|{fmt(r["reach_time_mean"])}|')
    lines += ['', '子集可能包含不同φ和不同样本数，不作为无偏排名；逐φ分母见per_phi.csv。没有可行样本或没有达到者的统计写NA。', '',
        '## 这20个val φ的结论', '',
        '共同时间下，概率贪心Pdet=.7461高于快速割草机.5606、概率/路程贪心.6464及保守割草机.1172，但低于ETO .7923和最近邻.7909。'
        '概率贪心达到ETO目标概率的平均时间9.037s，快速割草机9.808s，概率/路程版10.530s，保守割草机45.865s，四组都在120s内达到。'
        '本次路程归一化没有提高平均搜索指标；仅此20个val条件，不能推断其他分布、地图和相机参数。', '',
        '保守版条件Tfind仅.433s，同时只检测到.1172的概率质量，不能据此说它搜索更快。该条件统计只包含少量较早检测到的目标。'
        '引导加平滑缩放的diffusion只77/160条物理可行；仅引导160条都未通过全部动力学检查。全样本检测指标只是回放结果。'
        '保留的可行子集覆盖不同φ，不能与全20φ的传统策略直接作为公平排名。', '',
        '## 耗时口径与验证', '',
        f'共享可见图构建耗时{primitive["graph_setup_seconds"]:.6f}s，单独报告，不重复加进每条策略时间。割草机各只构建一次，表中列一次完整构建耗时；不是乘20，也不除20。'
        '贪心时间包含完整搜索路径生成、在线观测和决策，直到全部自由格子可见或120s上限；离线指标评测不计入。'
        '它们通常规划/模拟比Tφ更长的搜索过程，不能将表中时间理解为只计算Tφ前缀。ETO沿用含编译的历史planning_seconds_including_compile；NN和diffusion沿用历史生成/后处理耗时。'
        '计时范围与测量时段不同，仅作数量级参考，不宣称严格硬件测速优势。', '',
        '42条新增轨迹均检查完整线段、安全圆、边界、速度/加速度和Euler残差，并逐格验证在线首次观测与原harness回放一致。'
        '100条ETO在各自tf下逐格复现原first_seen_grid，随后才计算共同预算指标。固定路径末端一圈后不再增加观测；四组代表轨迹额外核对捷径回放与完整120s回放。', '',
        '## 文件', '',
        '- results.csv/json：520条结果（80条传统＋100 ETO＋20 NN＋160后处理diffusion＋160仅引导diffusion）。',
        '- references.csv：每φ的Tφ和目标Pφ；per_phi.csv：逐φ及可行子集统计；summary.csv/json：等权汇总、标准差、范围和中位数。',
        '- trajectories/：42条新增轨迹、位置/速度/控制、完整首见网格、目标选择事件和约束检查。',
        '- replays.npz：全部方法的轨迹及共同截止/120s的first_seen_grid。',
        '- classical_all_val.png、comparison_first6.png、detection_first6.png：固定按seed排序画全部20φ或前6φ，不按效果挑选。',
        '- PROTOCOL.json、primitive_checks.json、replay_checks.json、verification.json：固定设置及验证。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plots(data, rows, references, archived, env)


def plots(data, rows, references, archived, env):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    colors = dict(zip(ORDER, ['#0072B2', '#56B4E9', '#D55E00', '#E69F00', '#222222', '#009E73', '#CC79A7', '#888888']))
    seeds = sorted(set(r['phi_seed'] for r in rows))
    def background(ax, seed):
        i = int(np.flatnonzero(data['phi_seed'] == seed)[0])
        ax.imshow(data['phi_grid'][i], origin='lower', extent=(0,1,0,1), cmap='Greys', alpha=.5)
        for o in env.obstacles:
            ax.add_patch(Rectangle((o.xmin,o.ymin),o.xmax-o.xmin,o.ymax-o.ymin,fc='gray'))
        for cx,cy,r in env.collision_disks: ax.add_patch(Circle((cx,cy),r,fill=False,ec='red',ls=':'))
        ax.plot(.1,.1,'ko',ms=3)
        ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',title=f'val phi {seed}')
    def path(ax, row, label=True):
        key = f'{row["phi_seed"]}_{row["method"]}_{row["sample_id"]}'
        states = archived[key+'_states']
        node_times = np.arange(len(states))*row['tf']/len(states)
        horizon = row['common_time']
        times = np.r_[node_times[node_times < horizon], horizon]
        xy = np.column_stack([np.interp(times,node_times,states[:,j]) for j in (0,1)])
        ax.plot(xy[:,0],xy[:,1],c=colors[row['method']],alpha=.65,lw=1.3,
                label=ENGLISH[row['method']] if label else None)
    fig, axes = plt.subplots(4,5,figsize=(18,14),layout='constrained')
    for ax,seed in zip(axes.flat,seeds):
        background(ax,seed)
        for r in rows:
            if r['phi_seed'] == seed and r['method'] in ORDER[:4]: path(ax,r)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.04))
    fig.suptitle('Classical search prefixes at common ETO mean time; all val phi')
    fig.savefig(HERE/'classical_all_val.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for ax,seed in zip(axes.flat,seeds[:6]):
        background(ax,seed)
        for method in ORDER[:-1]:
            group = [r for r in rows if r['phi_seed']==seed and r['method']==method]
            for j,r in enumerate(group): path(ax,r,j==0)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.12))
    fig.suptitle('Common-time trajectories; all ETO and guided samples retained (including unsafe)')
    fig.savefig(HERE/'comparison_first6.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15,9),layout='constrained')
    times = np.linspace(0,CAP,2401)
    for ax,seed in zip(axes.flat,seeds[:6]):
        i = int(np.flatnonzero(data['phi_seed']==seed)[0]);weights=data['phi_grid'][i].ravel()
        ref = next(r for r in references if r['phi_seed']==seed)
        for method in ORDER[:-1]:
            curves=[]
            for r in rows:
                if r['phi_seed']!=seed or r['method']!=method:continue
                key=f'{seed}_{method}_{r["sample_id"]}_first_120'
                first=archived[key];order=np.argsort(first);cdf=np.r_[0.,np.cumsum(weights[order])]
                curves.append(cdf[np.searchsorted(first[order],times,side='right')])
            ax.plot(times,np.mean(curves,axis=0),c=colors[method],label=ENGLISH[method])
        ax.axvline(ref['common_time'],c='black',ls=':',lw=1)
        ax.axhline(ref['target_probability'],c='black',ls='--',lw=1)
        ax.set(title=f'val phi {seed}',xlabel='Time (s)',ylabel='Pdet',ylim=(0,1.02),xlim=(0,CAP))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.14))
    fig.savefig(HERE/'detection_first6.png',dpi=160,bbox_inches='tight');plt.close(fig)
