python main.py \
    algorithm=hsevo \
    model=gemini/gemini-2.5-flash \
    problem=op_sch \
    init_pop_size=2 \
    max_fe=20 \
    max_iter=10

# huggingface-cli download heurigen/heurigen-data --repo-type dataset --local-dir datasets