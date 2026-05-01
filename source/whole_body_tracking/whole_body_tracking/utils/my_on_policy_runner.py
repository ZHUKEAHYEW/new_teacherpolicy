from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _get_policy_and_normalizer(runner: OnPolicyRunner):
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
    """Local-only runner that exports ONNX checkpoints without cloud upload."""

    def save(self, path: str, infos=None):
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
