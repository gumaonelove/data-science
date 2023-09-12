from pydantic import BaseModel


class DialogueMessage(BaseModel):
    '''Сообщение в диалоге'''
    output: str