USD_RUB_RATE = 85
TOKEN_LIMIT = 1000
OPENAI_BIG = "gpt-4o-2024-08-06"
OPENAI_MINI = "gpt-4o-mini-2024-07-18"

prices_dict = {
        "gpt-3.5-turbo": {
            'output_tokens': 0.0000015,
            'prompt_tokens': 0.0000005
        },
        "gpt-4-turbo": {
            'output_tokens': 0.00003,
            'prompt_tokens': 0.00001
        },
        "gpt-4o": {
            'output_tokens': 0.000015,
            'prompt_tokens': 0.000005
        },
        "gpt-4o-2024-08-06": {
            'output_tokens': 0.00001,
            'prompt_tokens': 0.0000025
        },
        "gpt-4o-mini-2024-07-18": {
            'output_tokens': 0.0000006,
            'prompt_tokens': 0.00000015
        },
    }