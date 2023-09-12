from aiogram.dispatcher import FSMContext
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from config import TG_TOKEN
from messages import hello
from api import dialogue


# Initialize bot and dispatcher
bot = Bot(token=TG_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    name = message.from_user.first_name
    text = hello.format(name=name)
    await message.reply(text)


@dp.message_handler()
async def echo(message: types.Message, state: FSMContext):
    await message.answer('⌛️')

    async with state.proxy() as data:
        if 'messages' in data.keys():
            data['messages'].append(message.text)
        else:
            data['messages'] = [message.text]
        input = data['messages']

    ai_answer = await dialogue(input)

    async with state.proxy() as data:
        data['messages'].append(ai_answer)

        if len(data['messages']) == 6:
            data['messages'] = data['messages'][2:]

    await bot.edit_message_text(
        chat_id=message.chat.id, message_id=message.message_id + 1,
        text=ai_answer)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)