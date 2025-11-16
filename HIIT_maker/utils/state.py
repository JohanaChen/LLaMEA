_CHOICE_STATE = {
    "incumbent_id": None,
    "incumbent_render": None,
    "incumbent_json": None,
    "fitness_level": 1.0, # user preference score (for ABtest)
    "fitness_score": -float("inf"), # model-evaluated score (for LLMpredict)
    "printed_help": False,
    "last_feedback": None,
    "incumbent_feedback": None,
}
