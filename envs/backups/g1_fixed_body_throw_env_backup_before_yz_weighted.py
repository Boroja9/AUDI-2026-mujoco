
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

class G1FixedBodyThrowEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, xml_path=None, episode_time=1.8, control_dt=0.02, action_scale=0.5, joint_safety_margin=0.08, lock_right_hand=True, learned_release=False, scripted_release_time=0.60, target_pos=(0.45,0.0,0.01), success_radius=0.18):
        super().__init__()
        if xml_path is None: xml_path=Path(__file__).resolve().parents[1]/'assets'/'unitree_g1'/'scene_throw.xml'
        self.xml_path=Path(xml_path)
        self.model=mujoco.MjModel.from_xml_path(str(self.xml_path)); self.data=mujoco.MjData(self.model)
        self.episode_time=float(episode_time); self.control_dt=float(control_dt); self.frame_skip=max(1,int(round(self.control_dt/self.model.opt.timestep)))
        self.action_scale=float(action_scale); self.joint_safety_margin=float(joint_safety_margin); self.lock_right_hand=bool(lock_right_hand); self.learned_release=bool(learned_release); self.scripted_release_time=float(scripted_release_time); self.target_pos=np.array(target_pos,dtype=np.float64); self.success_radius=float(success_radius)
        self.ball_body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'throw_ball'); self.ball_geom_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'throw_ball_geom'); self.target_body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'throw_target'); self.hold_eq_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_EQUALITY,'hold_throw_ball'); self.ball_joint_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'throw_ball_free')
        self.target_geom_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'throw_target_geom'); self.stick_eq_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_EQUALITY,'stick_to_target')
        missing=[]
        if self.ball_body_id<0: missing.append('throw_ball')
        if self.target_body_id<0: missing.append('throw_target')
        if self.hold_eq_id<0: missing.append('hold_throw_ball')
        if self.ball_joint_id<0: missing.append('throw_ball_free')
        if missing: raise RuntimeError('Missing from G1 throwing scene: '+', '.join(missing)+'. Run scripts/create_g1_throw_scene.py first.')
        self.hold_body_id=int(self.model.eq_obj1id[self.hold_eq_id]); self.ball_qpos_adr=int(self.model.jnt_qposadr[self.ball_joint_id]); self.ball_qvel_adr=int(self.model.jnt_dofadr[self.ball_joint_id]); self.hold_relpose=self._load_hold_relpose()
        self.arm_joint_names=self._find_right_arm_joint_names()
        if not self.arm_joint_names: raise RuntimeError('Could not find right arm joints. Run scripts/inspect_g1.py and edit envs/g1_fixed_body_throw_env.py.')
        self.arm_joint_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.arm_joint_names]
        self.arm_qpos_adr=np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids]); self.arm_qvel_adr=np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.hand_joint_ids=np.array([
            joint_id for joint_id in range(self.model.njnt)
            if (name := mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_id))
            and name.startswith('right_hand_')
        ],dtype=np.int32)
        self.hand_qpos_adr=np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.hand_joint_ids],dtype=np.int32)
        self.hand_qvel_adr=np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.hand_joint_ids],dtype=np.int32)
        self.arm_joint_lower=self.model.jnt_range[self.arm_joint_ids,0]+self.joint_safety_margin
        self.arm_joint_upper=self.model.jnt_range[self.arm_joint_ids,1]-self.joint_safety_margin
        self.arm_actuator_ids=self._find_arm_actuator_ids(); self.n_arm=len(self.arm_joint_names)
        self.extra_actuator_ids=np.array([],dtype=np.int32); self.extra_joint_lower=np.array([]); self.extra_joint_upper=np.array([])
        self.action_space=spaces.Box(-1,1,shape=(self.n_arm+1,),dtype=np.float32)
        obs_dim=self.n_arm+self.n_arm+3+3+3+(self.n_arm+1)+1+1; self.observation_space=spaces.Box(-np.inf,np.inf,shape=(obs_dim,),dtype=np.float32)
        self.prev_action=np.zeros(self.n_arm+1); self.nominal_qpos=np.zeros(self.model.nq); self.nominal_ctrl=np.zeros(self.model.nu); self.locked_hand_qpos=None; self._init_nominal_pose()
        self.base_body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        self.nominal_base_height=float(self.data.xpos[self.base_body_id,2]) if self.base_body_id >= 0 else 0.0
        self.fall_height=0.55*self.nominal_base_height
        self.ball_radius=float(self.model.geom_size[self.ball_geom_id,0]); self.step_count=0; self.released=False; self.release_time=None; self.best_dist=np.inf; self.landing_error=None; self.success=False; self.robot_fell=False; self.stuck=False; self.ground_frozen=False
    def _find_right_arm_joint_names(self):
        all_names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(self.model.njnt)]; all_names=[n for n in all_names if n]
        preferred=['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']
        if all(n in all_names for n in preferred): return preferred
        candidates=[n for n in all_names if 'right' in n.lower() and any(k in n.lower() for k in ['shoulder','elbow','wrist','arm'])]
        ordered=[]
        for key in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']:
            for n in candidates:
                if key in n.lower() and n not in ordered: ordered.append(n)
        return ordered or candidates
    def _find_arm_actuator_ids(self):
        ids=[]
        for jname in self.arm_joint_names:
            jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,jname); found=-1
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid,0]==jid: found=aid; break
            if found<0: raise RuntimeError(f'Could not find actuator for joint {jname}. Run scripts/inspect_g1.py.')
            ids.append(found)
        return np.array(ids,dtype=np.int32)
    def apply_joint_target_offsets(self, offsets):
        """Add fixed radian offsets to position-actuated joints safely."""
        for joint_name, offset in offsets.items():
            joint_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
            if joint_id < 0:
                raise ValueError(f"Unknown joint: {joint_name}")
            actuator_id=-1
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid,0] == joint_id:
                    actuator_id=aid; break
            if actuator_id < 0:
                raise ValueError(f"No position actuator for joint: {joint_name}")
            lower=self.model.jnt_range[joint_id,0]+self.joint_safety_margin
            upper=self.model.jnt_range[joint_id,1]-self.joint_safety_margin
            self.nominal_ctrl[actuator_id]=np.clip(
                self.nominal_ctrl[actuator_id]+float(offset), lower, upper
            )
    def _init_nominal_pose(self):
        mujoco.mj_resetData(self.model,self.data)
        if self.model.nkey>0: mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data); self.nominal_qpos[:]=self.data.qpos[:]; self.nominal_ctrl[:]=0
        for aid in range(self.model.nu):
            trnid=self.model.actuator_trnid[aid,0]
            if trnid>=0:
                qadr=self.model.jnt_qposadr[trnid]
                if qadr<self.model.nq: self.nominal_ctrl[aid]=self.data.qpos[qadr]
        self.locked_hand_qpos=self.data.qpos[self.hand_qpos_adr].copy()

    def _lock_hand(self):
        if self.lock_right_hand and self.hand_qpos_adr.size:
            self.data.qpos[self.hand_qpos_adr]=self.locked_hand_qpos
            self.data.qvel[self.hand_qvel_adr]=0
    def _load_hold_relpose(self):
        root=ET.parse(self.xml_path).getroot(); weld=root.find("./equality/weld[@name='hold_throw_ball']")
        if weld is None: raise RuntimeError('Could not find hold_throw_ball weld in throwing scene XML.')
        relpose=np.fromstring(weld.attrib.get('relpose','0 0 0 1 0 0 0'),sep=' ',dtype=np.float64)
        if relpose.shape!=(7,): raise RuntimeError('hold_throw_ball relpose must have 7 numbers.')
        return relpose
    def _place_ball_in_hand(self):
        hand_pos=self.data.xpos[self.hold_body_id].copy(); hand_mat=self.data.xmat[self.hold_body_id].reshape(3,3)
        ball_pos=hand_pos+hand_mat@self.hold_relpose[:3]; ball_quat=np.empty(4,dtype=np.float64)
        mujoco.mju_mulQuat(ball_quat,self.data.xquat[self.hold_body_id],self.hold_relpose[3:7])
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr+3]=ball_pos; self.data.qpos[self.ball_qpos_adr+3:self.ball_qpos_adr+7]=ball_quat; self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6]=0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.model.nkey>0: mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        else: mujoco.mj_resetData(self.model,self.data); self.data.qpos[:]=self.nominal_qpos
        self.data.ctrl[:]=self.nominal_ctrl; self.data.qpos[self.arm_qpos_adr]+=self.np_random.uniform(-0.03,0.03,self.n_arm); self._lock_hand()
        if self.hold_eq_id>=0: self.data.eq_active[self.hold_eq_id]=1
        if self.stick_eq_id>=0: self.data.eq_active[self.stick_eq_id]=0
        self.model.geom_contype[self.ball_geom_id]=1; self.model.geom_conaffinity[self.ball_geom_id]=1
        mujoco.mj_forward(self.model,self.data); self._place_ball_in_hand(); self.model.body_pos[self.target_body_id]=self.target_pos; mujoco.mj_forward(self.model,self.data)
        self.step_count=0; self.released=False; self.release_time=None; self.best_dist=np.inf; self.landing_error=None; self.success=False; self.robot_fell=False; self.stuck=False; self.ground_frozen=False; self.prev_action=np.zeros(self.n_arm+len(self.extra_actuator_ids)+1)
        return self._get_obs(), {}
    def step(self, action):
        action=np.clip(np.asarray(action,dtype=np.float64),-1,1); self.data.ctrl[:]=self.nominal_ctrl
        arm_targets=self.nominal_ctrl[self.arm_actuator_ids]+self.action_scale*action[:self.n_arm]
        self.data.ctrl[self.arm_actuator_ids]=np.clip(arm_targets,self.arm_joint_lower,self.arm_joint_upper)
        n_extra=len(self.extra_actuator_ids)
        if n_extra:
            extra_targets=self.nominal_ctrl[self.extra_actuator_ids]+self.action_scale*action[self.n_arm:self.n_arm+n_extra]
            self.data.ctrl[self.extra_actuator_ids]=np.clip(extra_targets,self.extra_joint_lower,self.extra_joint_upper)
        t=self.step_count*self.control_dt
        if not self.released:
            do_release=action[-1]>0.5 if self.learned_release else t>=self.scripted_release_time
            if do_release:
                if self.hold_eq_id>=0: self.data.eq_active[self.hold_eq_id]=0
                self.released=True; self.release_time=t
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model,self.data)
            self._lock_hand()
        self.step_count+=1; ball_pos=self._ball_pos()
        # Poeni/pogodak se racunaju SAMO po X (dubina - koliko daleko je lopta
        # doletela u odnosu na metu), ne po Y (bocno) - Y namerno ne ulazi u
        # ovo rastojanje, robot za njega ne dobija niti gubi poene direktno.
        dist_3d=float(abs(ball_pos[0]-self.target_pos[0])); self.best_dist=min(self.best_dist,dist_3d)
        hit_target=False
        if self.released and not self.stuck and self.target_geom_id>=0:
            for i in range(self.data.ncon):
                c=self.data.contact[i]; g1,g2=int(c.geom1),int(c.geom2)
                if self.ball_geom_id in (g1,g2) and self.target_geom_id in (g1,g2):
                    hit_target=True; break
        if hit_target and self.stick_eq_id>=0:
            tpos=self.data.xpos[self.target_body_id]; tmat=self.data.xmat[self.target_body_id].reshape(3,3)
            relpos=tmat.T@(ball_pos-tpos); relquat=np.empty(4,dtype=np.float64)
            tquat_inv=np.empty(4,dtype=np.float64); mujoco.mju_negQuat(tquat_inv,self.data.xquat[self.target_body_id])
            mujoco.mju_mulQuat(relquat,tquat_inv,self.data.xquat[self.ball_body_id])
            self.model.eq_data[self.stick_eq_id,3:6]=relpos; self.model.eq_data[self.stick_eq_id,6:10]=relquat
            self.data.eq_active[self.stick_eq_id]=1
            self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6]=0
            self.model.geom_contype[self.ball_geom_id]=0; self.model.geom_conaffinity[self.ball_geom_id]=0
            self.stuck=True
        landed_on_ground=bool(self.released and ball_pos[2] <= self.ball_radius+0.015)
        # ZAMRZNI loptu cim prvi put dodirne zemlju (ako se vec nije zalepila
        # za metu) - inace se lopta kotrlja/odskace tokom preostalog dela
        # epizode (npr. dok cekamo da se ruka smiri) i zavrsi vizuelno daleko
        # od mesta gde je landing_error/success stvarno izmeren. Kolizija sa
        # podom OSTAJE aktivna (inace propada kroz pod) - samo joj se brzina
        # ponistava SVAKI korak dok je "na zemlji", da stvarno miruje tu.
        if landed_on_ground and not self.stuck:
            self.ground_frozen=True
        if self.ground_frozen and not self.stuck:
            self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6]=0
        landed=bool(hit_target or landed_on_ground)
        if landed and self.landing_error is None:
            self.landing_error=dist_3d; self.success=bool(dist_3d <= self.success_radius)
        self.robot_fell=bool(self.base_body_id >= 0 and self.data.xpos[self.base_body_id,2] < self.fall_height)
        obs=self._get_obs(); reward=self._compute_reward(action, landed); terminated=bool(landed or self.robot_fell); truncated=bool(self.step_count*self.control_dt>=self.episode_time)
        info={'dist_to_target':dist_3d,'best_dist':float(self.best_dist),'landing_error':self.landing_error,'success':self.success,'released':self.released,'release_time':self.release_time,'completion_time':self.step_count*self.control_dt if landed else None,'robot_fell':self.robot_fell,'arm_joint_names':self.arm_joint_names}
        self.prev_action=action.copy(); return obs,float(reward),terminated,truncated,info
    def _compute_reward(self, action, landed):
        dist_3d=np.linalg.norm(self._ball_pos()-self.target_pos)
        reward=-0.002*np.linalg.norm(action[:self.n_arm])-0.002*np.linalg.norm(action-self.prev_action)
        if self.robot_fell:
            return reward-5.0
        if landed:
            return reward + (10.0 if self.success else np.exp(-6.0*dist_3d))
        if self.released:
            reward += 0.05*np.exp(-4.0*dist_3d)
        return reward
    def _get_obs(self):
        return np.concatenate([self.data.qpos[self.arm_qpos_adr], self.data.qvel[self.arm_qvel_adr], self._ball_pos(), self._ball_vel(), self.target_pos, self.prev_action, [1.0 if self.released else 0.0], [max(0.0,self.episode_time-self.step_count*self.control_dt)]]).astype(np.float32)
    def _ball_pos(self): return self.data.xpos[self.ball_body_id].copy()
    def _ball_vel(self):
        vel=np.zeros(6); mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.ball_body_id,vel,0); return vel[:3].copy()
