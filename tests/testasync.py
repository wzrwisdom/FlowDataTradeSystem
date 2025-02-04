import asyncio
import threading

async def hello(name):
    # 打印name和当前线程:
    print("Hello %s! (%s)" % (name, threading.current_thread))
    # 异步调用asyncio.sleep(1):
    # await asyncio.sleep(1)
    await asyncio.sleep(1)
    # asyncio.sleep(2)
    print("Hello %s again! (%s)" % (name, threading.current_thread))
    return name

async def main():
    L = await asyncio.gather(hello("Bob"), hello("Alice"))
    # print(L)
    # await hello("Bob")
    # await hello("Alice")

asyncio.run(main())