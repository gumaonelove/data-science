import aiohttp
import ssl

from config import DIALOGUE_URL


async def dialogue(messages: list):

    async with aiohttp.ClientSession() as session:
        async with session.post(DIALOGUE_URL, json=messages) as response:
            ans = await response.json()

    return ans['output']