# LeggedLab 系统架构图

本文档基于当前项目结构生成，重点描述训练、播放、任务注册、环境、资产、Runner、WMP/AMP 扩展之间的关系。

## 总览

```mermaid
flowchart TB
    User[用户 / CLI] --> Train[legged_lab/scripts/train.py]
    User --> Play[legged_lab/scripts/play.py]
    User --> GUI[legged_lab/scripts/launcher_gui.py]

    Train --> App[IsaacLab AppLauncher]
    Play --> App
    GUI --> Train
    GUI --> Play

    Train --> Registry[legged_lab/utils/task_registry.py]
    Play --> Registry

    Registry --> TaskCfg[EnvCfg + AgentCfg]
    Registry --> EnvClass[EnvClass]

    TaskCfg --> BaseCfg[envs/base/base_env_config.py]
    BaseCfg --> SceneCfg[Scene / Sim / Robot / Terrain / Sensor 配置]
    BaseCfg --> TrainCfg[RSL-RL Agent 配置]

    EnvClass --> BaseEnv[envs/base/base_env.py<br/>BaseEnv]
    EnvClass --> RBEnv[envs/rb160w/rb160w_env.py<br/>RB160WEnv]

    SceneCfg --> Assets[legged_lab/assets<br/>USD / ArticulationCfg]
    SceneCfg --> Terrains[legged_lab/terrains<br/>plane / rough / WMP terrain]
    SceneCfg --> Sensors[IsaacLab Sensors<br/>Contact / RayCaster / Camera]

    BaseEnv --> Managers[CommandGenerator<br/>RewardManager<br/>EventManager]
    BaseEnv --> MDP[legged_lab/mdp/rewards.py]
    BaseEnv --> Obs[TensorDict Observations<br/>policy / critic]

    Obs --> DefaultRunner[rsl_rl OnPolicyRunner]
    Obs --> WMPRunner[legged_lab/runners/wmp_amp_runner.py<br/>WMPAMPRunner]

    DefaultRunner --> PPO[RSL-RL PPO]
    WMPRunner --> WMP[legged_lab/world_models/wmp<br/>RSSM World Model]
    WMPRunner --> AMP[legged_lab/amp<br/>Motion Loader / Discriminator / Replay]
    WMPRunner --> WMPAMPAlgo[legged_lab/algorithms/wmp_amp_ppo.py]

    PPO --> Logs[logs/<experiment_name><br/>checkpoints / params / exported policy]
    WMPAMPAlgo --> Logs
    Play --> Export[policy.pt / policy.onnx]
    Export --> Logs
```

## 任务注册关系

```mermaid
flowchart LR
    Init[legged_lab/envs/__init__.py] --> Registry[task_registry.register]

    Registry --> H1[h1_flat / h1_rough]
    Registry --> G1[g1_flat / g1_rough]
    Registry --> GR2[gr2_flat / gr2_rough]
    Registry --> B2[b2_flat / b2_rough]
    Registry --> B2RGBD[b2_rgbd_*]
    Registry --> WMPAMP[b2_rgbd_wmp_amp_*]
    Registry --> A1AMP[a1_amp_flat]
    Registry --> RB160W[rb160w_flat]

    RB160W --> RBEnv[RB160WEnv]
    RB160W --> RBEnvCfg[RB160WFlatEnvCfg]
    RB160W --> RBAgentCfg[RB160WFlatAgentCfg]

    B2RGBD --> BaseEnv[BaseEnv]
    WMPAMP --> BaseEnv
    H1 --> BaseEnv
    G1 --> BaseEnv
    GR2 --> BaseEnv
    B2 --> BaseEnv
    A1AMP --> BaseEnv
```

## 训练流程

```mermaid
sequenceDiagram
    actor U as 用户
    participant T as train.py
    participant A as AppLauncher
    participant R as task_registry
    participant E as EnvClass(BaseEnv/RB160WEnv)
    participant O as Runner
    participant L as logs

    U->>T: python train.py --task=...
    T->>A: 启动 Isaac Sim / IsaacLab
    T->>R: get_cfgs(task), get_task_class(task)
    R-->>T: env_cfg, agent_cfg, env_class
    T->>T: CLI 覆盖 num_envs / seed / runner
    T->>E: env_class(env_cfg, headless)
    E-->>T: VecEnv 接口
    T->>O: OnPolicyRunner 或 WMPAMPRunner
    T->>L: dump env.yaml / agent.yaml
    O->>E: act -> env.step()
    E-->>O: obs, reward, done, extras
    O->>O: PPO / AMP / WMP 更新
    O->>L: checkpoint / metrics
```

## 播放流程

```mermaid
sequenceDiagram
    actor U as 用户
    participant P as play.py
    participant R as task_registry
    participant E as Env
    participant Runner as Runner
    participant C as Checkpoint

    U->>P: python play.py --task=... --num_envs=...
    P->>R: 读取 env_cfg / agent_cfg
    P->>P: 关闭噪声、覆盖播放速度范围、设置环境数量
    P->>E: env_class(env_cfg, headless)
    P->>C: get_checkpoint_path()
    P->>Runner: load checkpoint
    Runner-->>P: inference policy
    P->>P: 导出 policy.pt / policy.onnx
    loop 仿真循环
        P->>Runner: policy(obs)
        Runner-->>P: actions
        P->>E: env.step(actions)
        E-->>P: obs, reward, done
    end
```

## 环境内部结构

```mermaid
flowchart TB
    Env[BaseEnv / RB160WEnv] --> Scene[IsaacLab InteractiveScene]
    Env --> Buffers[动作历史 / 观测历史 / reset buffer]
    Env --> Managers[Managers]

    Scene --> Robot[Articulation robot]
    Scene --> Terrain[Terrain / Ground Plane]
    Scene --> Contact[ContactSensor]
    Scene --> Ray[RayCaster / Height Scanner]
    Scene --> Camera[Gemini2 Depth/RGB Camera]

    Managers --> Command[CommandGenerator]
    Managers --> Reward[RewardManager]
    Managers --> Event[EventManager / Domain Randomization]

    Robot --> Obs[观测构造]
    Contact --> Obs
    Command --> Obs
    Ray --> Obs
    Camera --> WMPObs[WMP depth image]

    Obs --> PolicyObs[policy obs]
    Obs --> CriticObs[critic obs]
    WMPObs --> WMPRunner[WMPAMPRunner]
```

## 资产与机器人配置

```mermaid
flowchart LR
    AssetPy[legged_lab/assets/*/*.py] --> ArtCfg[ArticulationCfg]
    ArtCfg --> USD[USD Robot Asset]
    ArtCfg --> Init[init_state<br/>root pos / joint_pos]
    ArtCfg --> Actuator[ImplicitActuatorCfg<br/>stiffness / damping / limits]

    EnvCfg[Robot EnvCfg] --> ArtCfg

    Unitree[assets/unitree<br/>A1 / B2 / G1 / H1] --> AssetPy
    FFTAI[assets/fftai<br/>GR2] --> AssetPy
    Xuanji[assets/xuanji<br/>RB160W] --> AssetPy

    USD --> SceneRobot[env_cfg.scene.robot]
    Init --> SceneRobot
    Actuator --> SceneRobot
```

## WMP + AMP 扩展链路

```mermaid
flowchart TB
    Env[BaseEnv with RGBD] --> Depth[Gemini2 depth]
    Env --> Prop[get_wmp_proprioception]
    Env --> AMPObs[get_amp_observations]

    Depth --> Pre[depth_to_wmp_image<br/>64x64x1 NHWC]
    Pre --> WM[WorldModel RSSM]
    Prop --> WM
    WM --> Feature[WMP feature<br/>deter=512 / full=1536]

    Feature --> ObsAug[obs['wmp']]
    Env --> BaseObs[policy / critic obs]
    BaseObs --> ObsAug

    AMPObs --> Discriminator[AMPDiscriminator]
    Motion[AMP motion files<br/>datasets/wmp_mocap_motions] --> Loader[AMPLoader]
    Loader --> Discriminator

    ObsAug --> Algo[WMPAMPPPO]
    Discriminator --> AMPReward[AMP reward]
    AMPReward --> Algo
    Algo --> Rollout[RolloutStorage]
    WM --> Replay[WMPReplayBuffer<br/>replay_capacity]
    Replay --> WMTrain[World model train step]
```

## 关键目录职责

```text
legged_lab/scripts
  train.py / play.py / launcher_gui.py / inspect_* / preview_*

legged_lab/envs
  任务注册、各机器人 EnvCfg / AgentCfg、BaseEnv、RB160WEnv

legged_lab/assets
  机器人 USD 资产路径、初始姿态、执行器配置

legged_lab/mdp
  reward 函数和 MDP 相关计算

legged_lab/runners
  默认 rsl_rl 之外的 AMP / WMP-AMP runner

legged_lab/algorithms
  AMP-PPO、WMP-AMP-PPO 算法封装

legged_lab/amp
  AMP motion dataset、判别器、normalizer、replay buffer

legged_lab/world_models/wmp
  WMP/RSSM world model、encoder/decoder、replay buffer、depth preprocess

legged_lab/terrains
  rough terrain、ray caster、WMP terrain 相关配置和生成器

legged_lab/utils
  task_registry、CLI 参数、rsl_rl 版本兼容、键盘控制、scene 工具
```

