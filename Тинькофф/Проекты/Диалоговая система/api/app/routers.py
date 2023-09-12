from fastapi import APIRouter

from .dialogue import DialoBot
from .models import DialogueMessage

router = APIRouter()

bot = DialoBot()


@router.post('/dialogue/', response_model=DialogueMessage)
async def dialogue(messages: list) -> dict:
    '''Диалоговая система'''
    history = [i.capitalize() for i in messages]
    msg = bot(history)
    return {'output': msg}