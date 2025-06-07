from aiogram import Bot, Dispatcher

import handlers
import bot


async def main():
    dp = Dispatcher()
    dp.include_router(handlers.router)

    
    await dp.start_polling(bot.bot)

if __name__ == "__main__":
    import asyncio
    print('Солнце светит - бот Шарлотан пашет!')
    asyncio.run(main())