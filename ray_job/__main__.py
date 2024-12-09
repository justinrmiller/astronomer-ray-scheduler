import ray
import time

@ray.remote
def hello_world():
    return "hello world"

@ray.remote
def square(x):
    print(x)
    # time.sleep(100)
    return x * x

ray.init()  # No need to specify address="vertex_ray://...."
print(ray.get(hello_world.remote()))
print(ray.get([square.remote(i) for i in range(4)]))
