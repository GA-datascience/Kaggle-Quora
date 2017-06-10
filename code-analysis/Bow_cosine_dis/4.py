 df_train = gen_ngram_data(df_train)
    df_train = extract_counting_feat(df_train)
    df_train = extract_distance_feat(df_train)
    df_train = extract_tfidf_feat(df_train)
    df_test = gen_ngram_data(df_test)
    df_test = extract_counting_feat(df_test)
    df_test = extract_distance_feat(df_test)
    
    
    df_test = extract_tfidf_feat(df_test)
