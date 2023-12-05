from pypokerengine.players import BasePokerPlayer
from PokerBotSimple import PokerBotSimple

saved_model_to_test = 'Daniel_Negreanu'

def setup_ai():
    return PokerBotSimple(saved_model=saved_model_to_test, training_mode_on=False)

