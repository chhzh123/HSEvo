# fewer iter => num_function_eval=21
# python main.py \
#     algorithm=hsevo \
#     model=gemini/gemini-2.5-flash \
#     problem=tech_mapping \
#     init_pop_size=2 \
#     pop_size=2 \
#     max_fe=10 \
#     max_iter=10

# larger iter => num_function_eval=108
python main.py \
    algorithm=hsevo \
    model=gemini/gemini-2.5-flash \
    problem=tech_mapping \
    init_pop_size=2 \
    pop_size=10 \
    max_fe=100 \
    max_iter=10
