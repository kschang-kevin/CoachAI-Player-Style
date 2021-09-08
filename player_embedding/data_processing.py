import numpy as np
import pandas as pd
import random

target_players = ['Anders ANTONSEN',
 'Anthony Sinisuka GINTING',
 'CHEN Long',
 'CHEN Yufei',
 'CHOU Tien Chen',
 'Jonatan CHRISTIE',
 'Kento MOMOTA',
 'Khosit PHETPRADAB',
 'NG Ka Long Angus',
 'PUSARLA V. Sindhu',
 'SHI Yuqi',
 'TAI Tzu Ying',
 'Viktor AXELSEN',
 'WANG Tzu Wei']

# target_players = ['Anders ANTONSEN',
#  'Anthony Sinisuka GINTING',
#  'CHOU Tien Chen',
#  'Jonatan CHRISTIE',
#  'Kento MOMOTA',
#  'NG Ka Long Angus',
#  'Viktor AXELSEN']

def train_test_split(label2rids, test_ratio):
    test = random.sample(label2rids, k=round(len(label2rids)*test_ratio))
    train = [rid for rid in label2rids if rid not in test]
    return train, test

def player2cat(player):
    p2c = {'Anders ANTONSEN': 0, 'Anthony Sinisuka GINTING': 1, 'CHEN Long': 2,
     'CHEN Yufei': 3, 'CHOU Tien Chen': 4, 'Jonatan CHRISTIE': 5, 'Kento MOMOTA': 6,
     'Khosit PHETPRADAB': 7, 'NG Ka Long Angus': 8, 'PUSARLA V. Sindhu': 9,
     'SHI Yuqi': 10, 'TAI Tzu Ying': 11, 'Viktor AXELSEN': 12, 'WANG Tzu Wei': 13}
    return p2c[player]

# def player2cat(player):
#     p2c = {'Anders ANTONSEN': 0, 'Anthony Sinisuka GINTING': 1, 'CHOU Tien Chen': 2,
#        'Jonatan CHRISTIE': 3, 'Kento MOMOTA': 4, 'NG Ka Long Angus': 5,
#        'Viktor AXELSEN': 6}
#     return p2c[player]

def generate_labels(rally_data):
    # predict player A and B
    playerA = rally_data['name_A'].values[0]
    playerB = rally_data['name_B'].values[0]

    if playerA in target_players and playerB in target_players:
        return np.array([player2cat(playerA)]),  np.array([player2cat(playerB)])
    elif playerA not in target_players and playerB in target_players:
        return None,  np.array([player2cat(playerB)])
    elif playerA in target_players and playerB not in target_players:
        return np.array([player2cat(playerA)]),  None
    elif playerA in target_players and playerB in target_players:
        return None,  None
    
def type2cat(shot_type):
    t2c = {'發短球': 1, '長球': 2, '推撲球': 3, '殺球': 4, '接殺防守': 5, '平球': 6,
           '網前球': 7, '挑球': 8, '切球': 9, '發長球': 10}
    return t2c[shot_type]

# def type2cat(shot_type):
#     t2c = {'發短球': 0, '長球': 1, '推球': 2, '殺球': 3, '擋小球': 4, '撲球': 5,
#            '平球': 6, '放小球': 7, '挑球': 8, '點扣': 9, '勾球': 10, '過度切球': 11,
#            '防守回抽': 12, '防守回挑': 13, '發長球': 14, '切球': 15, '後場抽平球': 16,
#            '未知球種': 17, '小平球': 18}
    return t2c[shot_type]

def process_rally(rally_data):
    ## process config
    mean_x, std_x = 630., 160.
    mean_y, std_y = 470., 105.
    
    drop_cols = ['rally', 'match_id', 'set', 'rally_id', 'ball_round', 'time', 'frame_num', 'db', 'flaw', 'lose_reason', 'win_reason', 'type', 'server', # no need
                 'hit_area', 'landing_area', 'player_location_area', 'opponent_location_area', # area dup with x/y
                 'name_A', 'name_B', 'getpoint_player', 'roundscore_A', 'roundscore_B', # rally-wise features, maybe use later
                 'landing_height', 'landing_x', 'landing_y'] # landing info is dup with hitting
#     drop_cols = ['rally', 'ball_round', 'time', 'frame_num', 'db', 'flaw', 'lose_reason', 'win_reason', 'type', 'server', # no need
#                  'hit_area', 'landing_area', 'player_location_area', 'opponent_location_area', # area dup with x/y
#                  'name_A', 'name_B', 'getpoint_player', 'roundscore_A', 'roundscore_B', # rally-wise features, maybe use later
#                  'landing_height', 'landing_x', 'landing_y'] # landing info is dup with hitting

    ## Get player name for checking
    playerA = rally_data['name_A'].values[0]
    playerB = rally_data['name_B'].values[0]    
    
#     if rally_data['player'][0]
    
    ## process frame_num (time), get frame difference between last shot and this shot, 0 if serve ball 
    frame_diff = np.pad(rally_data['frame_num'].values[1:] - rally_data['frame_num'].values[:-1], (1, 0), mode='constant')
    rally_data['frame_diff'] = frame_diff
    
    ## NaN convert to binary
    rally_data['aroundhead'] = (rally_data['aroundhead'] == 1).astype(int)
    rally_data['backhand'] = (rally_data['backhand'] == 1).astype(int)
    
    ## Player A/B, convert to binary
    rally_data['player'] = (rally_data['player'] == 'A').astype(int)
    
    ## height convert to binary
    rally_data['hit_height'] = (rally_data['hit_height'] -1)
    rally_data['landing_height'] = (rally_data['landing_height'] -1)
    
    ## hit_x, hit_y fill with player location
    rally_data['hit_x'].values[0] = rally_data['player_location_x'].values[0]
    rally_data['hit_y'].values[0] = rally_data['player_location_y'].values[0]
    
    ## x/y standardization
    rally_data['hit_x'] = (rally_data['hit_x'] - mean_x)/std_x
    rally_data['hit_y'] = (rally_data['hit_y'] - mean_y)/std_y
    rally_data['landing_x'] = (rally_data['landing_x'] - mean_x)/std_x
    rally_data['landing_y'] = (rally_data['landing_y'] - mean_y)/std_y
    rally_data['player_location_x'] = (rally_data['player_location_x'] - mean_x)/std_x
    rally_data['player_location_y'] = (rally_data['player_location_y'] - mean_y)/std_y
    rally_data['opponent_location_x'] = (rally_data['opponent_location_x'] - mean_x)/std_x
    rally_data['opponent_location_y'] = (rally_data['opponent_location_y'] - mean_y)/std_y
    
    # type convert to category
    rally_data['type_code'] = [type2cat(t) for t in rally_data['type'].values]
    
    ## drop unneccesary columns
    rally_data.drop(columns=drop_cols, inplace=True)
    
    ## create a copy of the rally but with opposite player 

    if rally_data['player'][0] == 1:
        target_first = 1
        opponent_first = 0
    elif rally_data['player'][0] == 0:
        target_first = 0
        opponent_first = 1
    target_rally = rally_data.copy().loc[rally_data['player'] == 1].drop(columns=['player'], axis=1)
    opponent_rally = rally_data.copy().loc[rally_data['player'] == 0].drop(columns=['player'], axis=1)

    if playerA in target_players and playerB in target_players:
        return (target_rally.values, opponent_rally.values, target_first), (opponent_rally.values, target_rally.values, opponent_first)
    elif playerA not in target_players and playerB in target_players:
        return None, (opponent_rally.values, target_rally.values, opponent_first)
    elif playerA in target_players and playerB not in target_players:
        return (target_rally.values, opponent_rally.values, target_first), None
    elif playerA in target_players and playerB in target_players:
        return None,  None
    
def check_nan(np_rally):
    if np_rally is None:
        return False
    else:
        return np.isnan(np.sum(np_rally[0])) or np.isnan(np.sum(np_rally[1]))
