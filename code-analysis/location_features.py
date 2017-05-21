import re

locations = pd.read_csv("Movehub/cities.csv")


# Selects only the country column, drops any NA values and convert to a set 
countries = set(locations['Country'].dropna(inplace=False).values.tolist())
all_places = countries

# Turn it into a Regex
regex = "|".join(sorted(set(all_places)))


# Execution of the generating features
from tqdm import tqdm

## Generate features for the train set first

subset = df_train.shape[0] # Remove the subsetting 

results = []
print("processing:", df_train[0:subset].shape)
for index, row in tqdm(df_train[0:subset].iterrows()):
    q1 = str(row['question1'])
    q2 = str(row['question2'])

    rr = {}

    q1_matches = []
    q2_matches = []

    if (len(q1) > 0):
        q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]

    if (len(q2) > 0):
        q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]

    rr['z_q1_place_num'] = len(q1_matches)
    # rr['z_q1_has_place'] =len(q1_matches) > 0

    rr['z_q2_place_num'] = len(q2_matches) 
    # rr['z_q2_has_place'] = len(q2_matches) > 0

    rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))
    # rr['z_place_match'] = rr['z_place_match_num'] > 0

    rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))
    # rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0

    results.append(rr)     
    
train_locations = pd.DataFrame.from_dict(results)
    
## Generate features for the test set next

subset = df_test.shape[0] # Remove the subsetting 

results = []
print("processing:", df_test[0:subset].shape)
for index, row in tqdm(df_test[0:subset].iterrows()):
    q1 = str(row['question1'])
    q2 = str(row['question2'])

    rr = {}

    q1_matches = []
    q2_matches = []

    if (len(q1) > 0):
        q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]

    if (len(q2) > 0):
        q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]

    rr['z_q1_place_num'] = len(q1_matches)
    # rr['z_q1_has_place'] =len(q1_matches) > 0

    rr['z_q2_place_num'] = len(q2_matches) 
    # rr['z_q2_has_place'] = len(q2_matches) > 0

    rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))
    # rr['z_place_match'] = rr['z_place_match_num'] > 0

    rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))
    # rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0

    results.append(rr)     

test_locations = pd.DataFrame.from_dict(results)

# Write out these 2 datasets

train_locations.to_csv('train_locations.csv', index=False)
test_locations.to_csv('test_locations.csv', index=False)

# Concatenate into the x_train and x_test csv
#==============================================================================
# 
# # features = ['z_q1_place_num', 'z_q1_has_place', 'z_q2_place_num', 'z_q2_has_place',
#             'z_place_match_num', 'z_place_match', 
#             'z_place_mismatch_num', 'z_place_mismatch']
#==============================================================================

features = ['z_q1_place_num', 'z_q2_place_num', 'z_place_match_num', 'z_place_mismatch_num']

x_train[features] = train_locations[features]
x_test[features] = test_locations[features]

