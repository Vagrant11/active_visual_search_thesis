"""Render existing experiment artifacts; never import or invoke a planner."""
import argparse
import hashlib
import json
import sys
from contextlib import nullcontext, contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.lines import Line2D
from PIL import Image

from milestone_3.experiment import Environment, camera_trajectory
from visibility import RectangleObstacle, EPS, is_visible


class VideoWriter:
    """Use the already-installed OpenCV H.264 encoder, without installing tools."""
    def __init__(self, fps):
        self.fps = fps

    @contextmanager
    def saving(self, fig, path, dpi):
        import cv2
        self.fig, self.cv2 = fig, cv2
        width, height = fig.canvas.get_width_height()
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), self.fps, (width,height))
        if not self.writer.isOpened():
            raise RuntimeError('H.264 encoder could not open '+path)
        try:
            yield self
        finally:
            self.writer.release()

    def grab_frame(self):
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        self.writer.write(self.cv2.cvtColor(rgba,self.cv2.COLOR_RGBA2BGR))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_fan(pose, environment):
    """Display polygon: ray/rectangle clipping, with detector geometry from source."""
    x, y, theta = pose
    half = environment.camera.half_fov_radians
    angles = np.linspace(theta-half, theta+half, 361)
    # Include obstacle corners so display shadows have sharp edges.
    for o in environment.obstacles:
        for px, py in ((o.xmin,o.ymin),(o.xmin,o.ymax),(o.xmax,o.ymin),(o.xmax,o.ymax)):
            offset = (np.arctan2(py-y, px-x)-theta+np.pi) % (2*np.pi)-np.pi
            if -half <= offset <= half:
                angles = np.r_[angles, theta+offset-EPS, theta+offset, theta+offset+EPS]
    angles = np.sort(angles)
    endpoints = []
    for angle in angles:
        direction = (np.cos(angle), np.sin(angle))
        length = environment.camera.sensing_radius
        for o in environment.obstacles:
            enter, leave = 0., length
            hit = True
            for start, delta, lo, hi in ((x,direction[0],o.xmin,o.xmax),(y,direction[1],o.ymin,o.ymax)):
                if abs(delta) < EPS:
                    if start < lo or start > hi:
                        hit = False
                        break
                else:
                    a, b = sorted(((lo-start)/delta, (hi-start)/delta))
                    enter, leave = max(enter,a), min(leave,b)
                    if enter > leave:
                        hit = False
                        break
            if hit:
                length = min(length, max(0., enter))
        endpoints.append((x+length*direction[0], y+length*direction[1]))
    return np.array([(x,y), *endpoints, (x,y)])


def render(args):
    output = Path(__file__).resolve().parent
    if args.output_subdir:
        output = output/args.output_subdir
        output.mkdir(parents=True, exist_ok=True)
    formal = ROOT/'milestone_4/results/formal'
    config_path = formal/'config.json'
    config = json.loads(config_path.read_text())
    manifest = config['frozen_manifest']
    scene = next(s for s in manifest['candidate_manifest']['scenes'] if s['scene_id']==args.scene)
    inv = manifest['planner_and_evaluation_invariants']
    env = Environment(obstacles=tuple(RectangleObstacle(**o) for o in scene['obstacles']))
    assert env.camera.sensing_radius == inv['camera_range']
    assert env.camera.fov_degrees == inv['camera_fov_degrees']
    assert env.observation_dt == inv['observation_dt_seconds']
    assert env.budget == inv['budget_seconds']
    for name in ('milestone_3/experiment.py','milestone_2/visibility.py'):
        assert digest(ROOT/name)==config['source_sha256'][name], name
    input_path = formal/'inputs'/f'{args.scene}.npz'
    with np.load(input_path, allow_pickle=False) as data:
        points, truth, support = data['points'].copy(), data['oracle'].copy(), data['free_mask'].copy()
        priors = {key:data[key].copy() for key in args.priors}
        # Deterministic illustrative draw: first episode of the last frozen seed.
        seed = manifest['final_evaluation_sampling']['seeds'][-1]
        target_index = int(data[f'target_indices_seed_{seed}'][0])
        assert np.array_equal(data[f'targets_seed_{seed}'][0], points[target_index])
    assert int(np.random.default_rng(seed).choice(len(points),
               size=manifest['final_evaluation_sampling']['targets_per_seed'],p=truth)[0]) == target_index
    n = manifest['candidate_manifest']['grid_size']
    x_axis, y_axis = np.unique(points[:,0]), np.unique(points[:,1])
    assert len(x_axis)==len(y_axis)==n
    dx, dy = np.diff(x_axis)[0], np.diff(y_axis)[0]
    extent = (x_axis[0]-dx/2, x_axis[-1]+dx/2, y_axis[0]-dy/2, y_axis[-1]+dy/2)
    target = points[target_index]
    tracked = [config_path,input_path,ROOT/'milestone_3/experiment.py',ROOT/'milestone_2/visibility.py']
    prepared = []
    for prior_id in args.priors:
        label = next(p for p in config['expected_plans'] if p['scene_id']==args.scene
                     and p['gamma']==args.gamma and p['prior_id']==prior_id)
        directory = formal/'plans'/label['plan_id']
        result = json.loads((directory/'result.json').read_text())
        assert result['valid']
        for name in ('trajectory.npz','first_seen_grid.npz','exact_metrics.json'):
            assert digest(directory/name)==result['artifact_sha256'][name], name
        assert hashlib.sha256(priors[prior_id].tobytes()).hexdigest() == manifest['prior_sha256_by_scene'][args.scene][prior_id]
        with np.load(directory/'trajectory.npz',allow_pickle=False) as data:
            states, tf = data['states'].copy(), float(data['tf'])
        with np.load(directory/'first_seen_grid.npz',allow_pickle=False) as data:
            first = data['first_seen'].copy()
        camera = camera_trajectory(states,tf,env)
        probabilities = np.array([truth[first<=t].sum() for t in camera[:,0]])
        exact = json.loads((directory/'exact_metrics.json').read_text())
        assert np.isclose(probabilities[-1],exact['exact_detection_probability'],rtol=0,atol=EPS)
        observed = [t for t,x,y,theta in camera if is_visible((x,y,theta),tuple(target),list(env.obstacles),env.camera)]
        target_time = float(first[target_index])
        assert (observed and abs(observed[0]-target_time)<=EPS) or (not observed and np.isinf(target_time))
        tracked.extend(directory/name for name in ('trajectory.npz','first_seen_grid.npz','exact_metrics.json','result.json'))
        prepared.append((label,states,tf,first,camera,probabilities,target_time))
    before = {str(p.relative_to(ROOT)):digest(p) for p in tracked}
    prior_norms = {key:matplotlib.colors.Normalize(vmin=0.,vmax=float(p.max()))
                   for key,p in priors.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#e9eef6','axes.labelcolor':'#a8bbcc',
                         'xtick.color':'#a8bbcc','ytick.color':'#a8bbcc','font.size':12})
    metadata = {'scene_id':args.scene,'gamma':args.gamma,'no_planner_imported_or_run':True,
        'target':{'seed':seed,'episode':0,'grid_index':target_index,'xy':target.tolist(),
                  'selection':'First existing draw of last frozen evaluation seed; shared across videos'},
        'parameter_sources':{'camera':'milestone_3.experiment.Environment',
           'execution':'milestone_3.experiment.camera_trajectory',
           'obstacles_grid_sampling':'results/formal/config.json embedded frozen_manifest',
           'background':'results/formal/inputs/'+input_path.name+': oracle',
           'planner_prior_inset':'results/formal/inputs/'+input_path.name+': condition prior_id (verified against frozen SHA256)',
           'detection':'saved first_seen_grid.npz; probability = sum(oracle[first_seen <= t])'},
        'prior_inset':{'title':'Prior given to planner φ̂','colormap':'magma',
                       'placement':'right information column, above detection-probability curve',
                       'shared_across_videos':False,
                       'scales_by_prior':{key:{'vmin':norm.vmin,'vmax':norm.vmax}
                                          for key,norm in prior_norms.items()}},
        'camera':{'radius':env.camera.sensing_radius,'fov_degrees':env.camera.fov_degrees,
                  'scan_rate':env.scan_rate,'observation_dt':env.observation_dt,'budget':env.budget},
        'source_sha256_before':before,'videos':[]}
    previews=[]
    for label,states,tf,first,camera,probabilities,target_time in prepared:
        prior_id = label['prior_id']
        prior_norm = prior_norms[prior_id]
        path = output/f'{label["plan_id"]}.mp4'
        if path.exists():
            raise FileExistsError(path)
        fps = len(camera)/args.seconds
        detection_frame = int(np.flatnonzero(camera[:,0]>=target_time)[0]) if np.isfinite(target_time) else None
        fig = plt.figure(figsize=(14.4,9),dpi=100,facecolor='#101a28')
        ax=fig.add_axes([.065,.105,.50,.80],facecolor='#101a28')
        title = 'ORACLE' if prior_id=='oracle' else 'DIFFUSE / BLUR'
        fig.text(.065,.954,title,fontsize=25,weight='bold')
        level = 'true prior supplied to planner' if label['target_js_nats']==0 else f'JS = {label["target_js_nats"]:g} nats'
        fig.text(.065,.916,f'{args.scene}   |   gamma = {label["gamma"]:g}   |   {level}',fontsize=12,color='#a8bbcc')
        im=ax.imshow(truth.reshape(n,n),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=truth.max(),interpolation='nearest')
        overlay=np.zeros((n,n,4))
        overlay[:,:,:3]=matplotlib.colors.to_rgb('#20dccc')
        seen_image=ax.imshow(overlay,origin='lower',extent=extent,interpolation='nearest',zorder=2)
        for o in env.obstacles:
            ax.add_patch(Rectangle((o.xmin,o.ymin),o.xmax-o.xmin,o.ymax-o.ymin,facecolor='#8b98a6',edgecolor='white',lw=1.2,zorder=7,hatch='////'))
        ax.plot(states[:,0],states[:,1],color='#b4c4d7',alpha=.7,lw=1.6,ls='--',zorder=3)
        trail,=ax.plot([],[],color='#ffac54',lw=2.5,zorder=4)
        fan=Polygon(visible_fan(camera[0,1:],env),facecolor='#80cfff',alpha=.28,edgecolor='#b5e6ff',lw=1.5,zorder=5)
        ax.add_patch(fan)
        camera_dot,=ax.plot([],[],'o',color='#e8f8ff',markeredgecolor='#1476b8',markeredgewidth=2,ms=10,zorder=9)
        heading,=ax.plot([],[],color='white',lw=2,zorder=8)
        target_dot,=ax.plot(*target,marker='*',ms=19,color='#ef8dff',markeredgecolor='white',markeredgewidth=1.2,zorder=11)
        ring=Circle(target,env.camera.sensing_radius*.18,fill=False,ec='#52f2b3',lw=2.2,zorder=10,visible=False)
        ax.add_patch(ring)
        ax.annotate('sampled target',target,xytext=(12,13),textcoords='offset points',color='white',fontsize=10,zorder=12)
        for xy,word in ((states[0,:2],'START'),(states[-1,:2],'END')):
            ax.plot(*xy,'s',ms=5,color='white',zorder=8)
            ax.annotate(word,xy,xytext=(8,-16),textcoords='offset points',fontsize=9,color='white')
        ax.set(xlim=extent[:2],ylim=extent[2:],ylabel='y / workspace units')
        for spine in ax.spines.values():spine.set_color('#62758c')
        cax=fig.add_axes([.065,.06,.50,.012])
        cb=fig.colorbar(im,cax=cax,orientation='horizontal')
        cb.ax.tick_params(labelsize=9)
        fig.text(.065,.024,'Background: true probability mass per grid cell (same scale in both videos)',fontsize=10,color='#a8bbcc')
        side=.64
        fig.text(side,.855,'SIMULATION TIME',fontsize=11,color='#92a9c0')
        time_text=fig.text(side,.810,'',fontsize=22,weight='bold')
        fig.text(.807,.855,'DETECTED PROBABILITY',fontsize=10,color='#92a9c0')
        mass_text=fig.text(.807,.810,'',fontsize=24,weight='bold',color='#47e4ce')
        fig.text(side,.775,'P_det = true-prior mass of cells seen so far',fontsize=10,color='#a8bbcc')
        fig.text(side,.728,f'SHARED TARGET  /  seed {seed}, episode 0',fontsize=11,color='#92a9c0')
        fig.text(side,.696,f'x = {target[0]:.4f}   y = {target[1]:.4f}',fontsize=13)
        target_text=fig.text(side,.655,'',fontsize=18,weight='bold')
        event_text=fig.text(side,.625,'',fontsize=11,color='#52f2b3')
        event_frame_text=fig.text(side,.597,'',fontsize=9,color='#a8bbcc')
        # Separate figure axes keep the entire workspace, END and path unobscured.
        fig.text(side,.566,r'Prior given to planner $\hat{\phi}$',fontsize=13,color='#e9eef6')
        prior_ax=fig.add_axes([.725,.342,.135,.216],facecolor='#101a28')
        prior_im=prior_ax.imshow(priors[prior_id].reshape(n,n),origin='lower',extent=extent,
                                cmap='magma',norm=prior_norm,interpolation='nearest')
        prior_ax.set(xticks=[],yticks=[])
        for spine in prior_ax.spines.values():spine.set_color('#62758c')
        prior_cax=fig.add_axes([.725,.321,.135,.009])
        prior_cb=fig.colorbar(prior_im,cax=prior_cax,orientation='horizontal',
                             ticks=[prior_norm.vmin,prior_norm.vmax])
        prior_cb.ax.tick_params(labelsize=8,length=2,pad=1)
        prior_cb.ax.set_xticklabels([f'{prior_norm.vmin:g}',f'{prior_norm.vmax:.3g}'])
        fig.text(side,.286,'Independent colour scale / probability per cell',fontsize=10,color='#a8bbcc')
        chart=fig.add_axes([side,.13,.305,.125],facecolor='#152235')
        chart.set(xlim=(camera[0,0],env.budget),ylim=(0,1),xlabel='Simulation time / s',ylabel='P detected')
        chart.tick_params(labelsize=9)
        chart.xaxis.label.set_size(10);chart.yaxis.label.set_size(10)
        chart.grid(alpha=.12,color='white')
        for spine in chart.spines.values():spine.set_color('#3e536b')
        curve,=chart.plot([],[],color='#47e4ce',lw=2)
        cursor,=chart.plot([],[],'o',color='#47e4ce',ms=5)
        legend=[Line2D([0],[0],color='#b4c4d7',ls='--',label='Saved plan'),
                Line2D([0],[0],color='#ffac54',lw=2,label='Executed path'),
                Rectangle((0,0),1,1,fc='#20dccc',alpha=.6,label='Cells already seen'),
                Rectangle((0,0),1,1,fc='#80cfff',alpha=.5,label=f'{env.camera.fov_degrees:g} deg FoV, occlusion clipped')]
        fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(side-.005,.021),frameon=False,fontsize=10,ncol=2,columnspacing=1.1,handlelength=1.6)
        fig.text(side,.900,f'Budget {env.budget:g} s  |  observation step {env.observation_dt:g} s\nSaved plan tf = {tf:.3f} s  |  video = {args.seconds:g} s',fontsize=10,color='#a8bbcc',linespacing=1.5)
        fig.text(.065,.006,'Stored-plan replay  /  no replanning  /  coverage continues after this illustrative target is found',fontsize=10,color='#92a9c0')
        writer=VideoWriter(fps=fps)
        snapshots={0,len(camera)-1}
        if detection_frame is not None:snapshots.update((max(0,detection_frame-1),detection_frame))
        with (nullcontext() if args.preview_only else writer.saving(fig,str(path),dpi=100)):
            for frame,(t,x,y,theta) in enumerate(camera):
                if args.preview_only and frame not in snapshots:
                    continue
                seen=(first<=t).reshape(n,n)
                overlay[:,:,3]=seen*.48
                seen_image.set_data(overlay)
                # Include original knots to avoid shortcut chords in the displayed trail.
                node_times=np.arange(len(states))*tf/len(states)
                completed=states[node_times<=t,:2]
                executed=np.vstack((completed,[x,y]))
                trail.set_data(executed[:,0],executed[:,1])
                fan.set_xy(visible_fan((x,y,theta),env))
                camera_dot.set_data([x],[y])
                heading.set_data([x,x+env.camera.sensing_radius*.28*np.cos(theta)],[y,y+env.camera.sensing_radius*.28*np.sin(theta)])
                found=t>=target_time
                target_dot.set_color('#52f2b3' if found else '#ef8dff')
                ring.set_visible(found)
                time_text.set_text(f'{t:05.2f} / {env.budget:g} s')
                mass_text.set_text(f'{probabilities[frame]*100:05.2f}%')
                target_text.set_text(f'T_find = {target_time:.2f} s' if found else ('TIMEOUT / not detected' if frame==len(camera)-1 else 'Target not yet detected'))
                target_text.set_color('#52f2b3' if found else '#ef8dff')
                event_text.set_text('FIRST DETECTION' if frame==detection_frame else ('Detected at marked frame' if found else 'Range + FoV + clear line of sight'))
                event_frame_text.set_text(f'First frame: {detection_frame+1:03d} / {len(camera)}  |  video time {detection_frame/fps:.2f} s' if found else '')
                curve.set_data(camera[:frame+1,0],probabilities[:frame+1])
                cursor.set_data([t],[probabilities[frame]])
                if not args.preview_only:
                    writer.grab_frame()
                if frame in snapshots:
                    snapshot=output/f'{prior_id}__frame_{frame:03d}.png'
                    fig.savefig(snapshot,dpi=100,facecolor=fig.get_facecolor())
                    if frame in (detection_frame,len(camera)-1):previews.append(snapshot)
                if frame%75==0:print(f'{prior_id}: {frame+1}/{len(camera)} frames',flush=True)
        plt.close(fig)
        if args.preview_only:
            continue
        metadata['videos'].append({'file':path.name,'prior_id':prior_id,'frames':len(camera),'fps':fps,
            'duration_seconds':len(camera)/fps,'tf':tf,'target_t_find':target_time if np.isfinite(target_time) else None,
            'first_detection_frame_zero_based':detection_frame,'first_detection_frame_one_based':None if detection_frame is None else detection_frame+1,
            'first_detection_video_seconds':None if detection_frame is None else detection_frame/fps,
            'final_detection_probability':float(probabilities[-1]),'sha256':digest(path)})
        print(f'Finished {path.name}',flush=True)
    after={str(p.relative_to(ROOT)):digest(p) for p in tracked}
    assert before==after, 'A source artifact changed during rendering'
    assert 'milestone_3.planner' not in sys.modules
    metadata['source_sha256_after']=after
    metadata['source_files_unchanged']=True
    if not args.preview_only:
        (output/'render_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    canvas=Image.new('RGB',(1440,900*2),'#101a28')
    for i,p in enumerate(previews):
        with Image.open(p) as im:canvas.paste(im.resize((720,450)),((i%2)*720,(i//2)*450))
    canvas.crop((0,0,1440,900)).save(output/'preview.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',required=True)
    parser.add_argument('--gamma',type=float,required=True)
    parser.add_argument('--priors',nargs='+',required=True)
    parser.add_argument('--seconds',type=float,required=True)
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--output-subdir')
    render(parser.parse_args())
