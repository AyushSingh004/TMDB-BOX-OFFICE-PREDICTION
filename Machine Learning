columns_for_training = ["log_budget", "log_popularity", "log_runtime", "day_of_week", "year", "month", "week_of_year", "season",
                        "num_genres", "num_of_production_countries", "log_num_of_cast", "log_num_of_male_cast", "log_num_of_female_cast", "has_collection", 
                        "has_homepage", "has_tag", "is_english_language",
                       "log_num_of_crew", "log_num_of_male_crew", "log_num_of_female_crew",
                       "log_title_len", "log_overview_len", "log_tagline_len",
                       "log_num_of_directors", "log_num_of_producers", "log_num_of_editors", "log_num_of_art_crew", "log_num_of_sound_crew",
                       "log_num_of_costume_crew", "log_num_of_camera_crew", "log_num_of_visual_effects_crew", "log_num_of_lighting_crew",
                        "log_num_of_other_crew"]


 # adding isTopGenre_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('isTopGenre_'), axis=1).columns.values)
 
 # adding isTopProductionCompany_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('isTopProductionCompany_'), axis=1).columns.values)
 
 # adding isTopProductionCountry_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('isTopProductionCountry_'), axis=1).columns.values)
 
 # adding has_top_actor_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_actor_'), axis=1).columns.values)
 
 # adding has_top_keyword_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_keyword_'), axis=1).columns.values)
 
 # adding has_top_director_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_director_'), axis=1).columns.values)
 
 # adding has_top_producer_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_producer_'), axis=1).columns.values) 




.......Still need to complete
