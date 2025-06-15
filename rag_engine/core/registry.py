# Placeholder for dynamic component loader/registry
class Registry:
    def __init__(self):
        self.components = {}
    def register(self, name, component):
        self.components[name] = component
    def get(self, name):
        return self.components.get(name)
