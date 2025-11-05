class Configurable:
    """
    Instances of this class will maintain a config system, which is protected from unexpected modification
    """

    def __init__(self, config: dict = None):
        # initialize and specify the value in config
        self._config = config.copy()

    def get_config(self, copy=True) -> dict:
        """
        Return self._config
        :param copy:
        :return: a copy of config dict
        """
        if copy:
            return self._config.copy()
        return self._config

    def update_config(self, config: dict):
        """
        Merge config and self._config
        """
        self._config.update(config)

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        if self._config is not None:
            self._config = None

    @property
    def config(self):
        return self._config
