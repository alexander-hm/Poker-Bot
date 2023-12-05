from pypokerengine.players import BasePokerPlayer
from PokerBotSimple import PokerBotSimple
from pypokerengine.api.game import setup_config, start_poker

saved_model_to_test = "./models/Daniel_Negreanu"


def setup_ai():
    return PokerBotSimple()


