from sweagent.environment.hooks.abstract import EnvHook  
  
class OneRepoOneModelOneTimePatchHook(EnvHook):  
    def __init__(self, patch_data: str, repo_name: str):  
        self.patch_data = patch_data  
        self.repo_name = repo_name  
        self._applied = False  

    def on_init(self, *, env):  
        self._env = env  # Store the environment reference here  

    def on_environment_startup(self):  
        if not self._applied:  
            self._apply_patch_once()  
            self._applied = True  

    def _apply_patch_once(self):
        try:
            self._env.communicate(
                f"cd /{self.repo_name} && "
                "git config user.email 'zhengyanshi@microsoft.com' && "
                "git config user.name  'Zhengyan Shi'",
                check="raise",
            )
            cmd = (
                "cat <<'PATCH' > /tmp/one_time.patch\n"
                f"{self.patch_data}\n"
                "PATCH"
            )
            self._env.communicate(cmd, check="raise")
            self._env.communicate(f"cd /{self.repo_name} && git apply /tmp/one_time.patch", check="raise")  
            
            self._env.communicate(f"cd /{self.repo_name} && git add -A && git commit -m 'Applied one-time patch'", check="raise")
        except Exception as e:
            raise RuntimeError(f"Failed to apply one-time patch: {e}") from e