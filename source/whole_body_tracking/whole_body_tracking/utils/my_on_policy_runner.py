"""RSL-RL OnPolicyRunner 的小工具。

保留这个文件是为了兼容较早的训练/导出流程。当前 train/play 脚本主要直接使用
RSL-RL 自带的 `OnPolicyRunner`。
"""

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _get_policy_and_normalizer(runner: OnPolicyRunner):
    """返回 actor-critic 模块，以及存在时的 observation normalizer。"""
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    return policy_nn, normalizer


class MotionOnPolicyRunner(OnPolicyRunner):
    """只用于本地的 runner：保存 checkpoint 后导出 ONNX，不上传云端。"""

    def save(self, path: str, infos=None):
        """保存 checkpoint，然后在 checkpoint 旁边导出 ONNX policy。"""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        policy_nn, normalizer = _get_policy_and_normalizer(self)
        export_motion_policy_as_onnx(
            self.env.unwrapped,
            policy_nn,
            normalizer=normalizer,
            path=policy_path,
            filename=filename,
        )
        attach_onnx_metadata(self.env.unwrapped, "local", path=policy_path, filename=filename)
