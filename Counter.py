class Singleton(type):
    _instances = {}

    def __call__(cls, instance_id, *args, **kwargs):
        if instance_id not in cls._instances:
            instance = super().__call__(instance_id, *args, **kwargs)
            cls._instances[instance_id] = instance
        return cls._instances[instance_id]


class Counter(metaclass=Singleton):
    def __init__(self, instance_id, initial_value=0):
        self.counter = initial_value
        self.instance_id = instance_id

    def incrementCounter(self):
        self.counter += 1

    def resetCounter(self):
        self.counter = 0

    def getCounter(self):
        return self.counter

    def getInstanceId(self):
        return self.instance_id
