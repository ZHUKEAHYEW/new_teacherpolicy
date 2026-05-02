"""机器人环境任务实现包。"""

from isaaclab_tasks.utils import import_packages

##
# 注册 Gym 环境。
##


# blacklist 用于避免从子包导入配置。
_BLACKLIST_PKGS = ["utils"]
# 导入当前包中的所有配置。
import_packages(__name__, _BLACKLIST_PKGS)
