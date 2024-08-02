def crop_type(df, crop):
    df = df[(df['desc'] == crop)]
    return df


def season_filter(df, season):
    df = df[(df['season'] == season)]
    return df


def region_filter(df, region):
    df = df[(df['region'] == region)]
    return df


def season_specific_weight(df, season):
    df.loc[df['season'] == season, 'season_encoded'] += 1
    df.loc[df['season'] == season, 'season_encoded'] *= 10
    return df


def region_season_specific_weight(df, region):
    df.loc[df['region'] == region, 'season_encoded'] += 1
    df.loc[df['region'] == region, 'season_encoded'] *= 10
    return df


def season_specific_weightD(df, season):
    df.loc[df['season'] == season, 'season_encoded'] /= 10
    df.loc[df['season'] == season, 'season_encoded'] -= 1
    return df


def region_season_specific_weightD(df, region):
    df.loc[df['region'] == region, 'season_encoded'] /= 10
    df.loc[df['region'] == region, 'season_encoded'] -= 1
    return df
