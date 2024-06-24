#from pypokerengine.players import BasePokerPlayer
from PokerBotSimple import PokerBotSimple
from dumb_bots.honestbot import HonestBot
#from pypokerengine.api.game import setup_config, start_poker

saved_model_to_test = "./models/Daniel_Negreanu"


def setup_ai(): 
    #return HonestBot()
    return PokerBotSimple(saved_model=saved_model_to_test, training_mode_on=False)


