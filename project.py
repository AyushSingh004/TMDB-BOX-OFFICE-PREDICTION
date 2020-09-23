import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset=pd.read_csv('test.csv')
dataset1=pd.read_csv('train.csv')

dataset1.isna().sum()
dataset.isna().sum()

a=dataset1.head()
b=dataset1.describe()

dataset1.boxplot()

dataset1.hist()

dataset1.info()

dataset1=dataset1.select_dtypes(exclude=['object'])

#Genres

def genres_preprocessing(elem):
    string = str(elem)
    str1 = string.replace(']','').replace('[','').replace('{','').replace('}','').replace('\'','').replace(' ','').replace("name", "").replace("id", "").replace(":", "")
    ll = str1.split(",")[1::2]
    return ll

dataset1["genres_processed"] = dataset1.genres.apply(lambda elem: genres_preprocessing(elem))
dataset["genres_processed"] = dataset.genres.apply(lambda elem: genres_preprocessing(elem))

genres_dict = dict()

for genre in dataset1["genres_processed"]:
    for elem in genre:
        if elem not in genres_dict:
            genres_dict[elem] = 1
        else:
            genres_dict[elem] += 1

sns.set(rc={'figure.figsize':(12,8)})
genres_df = pd.DataFrame.from_dict(genres_dict, orient='index')
genres_df.columns = ["number_of_movies"]
genres_df = genres_df.sort_values(by="number_of_movies", ascending=False)
genres_df.plot.bar()
plt.title("Number of films per genre")

genres_df.index.values
for g in genres_df.index.values:
    dataset1['isGenre_' + g] = dataset1['genres_processed'].apply(lambda x: 1 if g in x else 0)
    dataset['isGenre_' + g] = dataset['genres_processed'].apply(lambda x: 1 if g in x else 0)

#English Language

dataset1.original_language.value_counts()[:10].plot.bar()
plt.title("Number of films per language")

dataset1["is_english_language"] = dataset1.original_language.apply(lambda x: 1 if x == "en" else 0)
dataset["is_english_language"] = dataset.original_language.apply(lambda x: 1 if x == "en" else 0)

dataset1.is_english_language = dataset1.is_english_language.fillna(1)
dataset.is_english_language = dataset.is_english_language.fillna(1)


#Production

def production_companies_preprocessing(elem):
    string = str(elem)
    str1 = string.replace(']','').replace('[','').replace('{','').replace('}','').replace(' ','').replace("name", "").replace("id", "").replace(":", "").replace("\'", "")
    ll = str1.split(",")[0::2]
    return ll

dataset1["production_companies"] = dataset1.production_companies.fillna('NoProductionCompany')
dataset["production_companies"] = dataset.production_companies.fillna('NoProductionCompany')

dataset1["production_companies_processed"] = dataset1.production_companies.apply(lambda elem: production_companies_preprocessing(elem))
dataset["production_companies_processed"] = dataset.production_companies.apply(lambda elem: production_companies_preprocessing(elem))



production_companies_dict = dict()

for production_company in dataset1["production_companies_processed"]:
    for elem in production_company:
        if elem not in production_companies_dict:
            production_companies_dict[elem] = 1
        else:
            production_companies_dict[elem] += 1


sns.set(rc={'figure.figsize':(12,8)})
production_companies_df = pd.DataFrame.from_dict(production_companies_dict, orient='index')
production_companies_df.columns = ["number_of_movies"]
production_companies_df = production_companies_df.sort_values(by="number_of_movies", ascending=False)
production_companies_df.head(10).plot.bar()
plt.title("Number of films per production company")

dataset1["num_of_production_companies"] = dataset1.production_companies_processed.apply(len)
dataset["num_of_production_companies"] = dataset.production_companies_processed.apply(len)

dataset1["num_of_production_companies"].value_counts().plot.bar()
plt.title("Number of multiple production companies per movie")


for i in production_companies_df.index.values:
    dataset1['isProductionCompany_' + i] = dataset1['production_companies_processed'].apply(lambda x: 1 if i in x else 0)
    dataset['isProductionCompany_' + i] = dataset['production_companies_processed'].apply(lambda x: 1 if i in x else 0)


#production Countries
    
    
def production_countries_preprocessing(elem):
    string = str(elem)
    str1 = string.replace(']','').replace('[','').replace('{','').replace('}','').replace(' ','').replace("name", "").replace("iso_3166_1", "").replace(":", "").replace("\'", "")
    ll = str1.split(",")[0::2]
    return ll

dataset1["production_countries_processed"] = dataset1.production_countries.fillna("NaN").apply(lambda elem: production_countries_preprocessing(elem))
dataset["production_countries_processed"] = dataset.production_countries.fillna("NaN").apply(lambda elem: production_countries_preprocessing(elem))


production_countries_dict = dict()

for production_country in dataset1["production_countries_processed"]:
    for elem in production_country:
        if elem not in production_countries_dict:
            production_countries_dict[elem] = 1
        else:
            production_countries_dict[elem] += 1

production_countries_df = pd.DataFrame.from_dict(production_countries_dict, orient='index')
production_countries_df.columns = ["number_of_movies"]
production_countries_df = production_countries_df.sort_values(by="number_of_movies", ascending=False)

for i in production_countries_df.index.values:
    dataset1['isProductionCountry_' + i] = dataset1['production_countries_processed'].apply(lambda x: 1 if i in x else 0)
    dataset['isProductionCountry_' + i] = dataset['production_countries_processed'].apply(lambda x: 1 if i in x else 0)


#popularity
    
f, ax = plt.subplots(3, figsize=(11,7))
sns.boxplot(x=dataset1.popularity, ax = ax[0])
ax[0].set_title("Popularity Boxplot")
sns.distplot(a=dataset1.popularity, kde = False, ax = ax[1])
ax[1].set_title("Popularity Histogram")
sns.distplot(a=np.log1p(dataset1.popularity), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed Popularity Histogram")
f.tight_layout()

dataset1["log_popularity"] = np.log1p(dataset1.popularity)
dataset["log_popularity"] = np.log1p(dataset.popularity)


#runtime

dataset1["runtime"] = dataset1["runtime"].fillna(dataset1["runtime"].mode()[0])
dataset["runtime"] = dataset["runtime"].fillna(dataset["runtime"].mode()[0])

dataset1.runtime = dataset1.runtime.fillna(dataset1.runtime.mode())

f, ax = plt.subplots(4, figsize=(12,7))
sns.boxplot(x=dataset1.runtime, ax = ax[0])
ax[0].set_title("Runtime Boxplot")
sns.distplot(a=dataset1.runtime, kde = False, ax = ax[1])
ax[1].set_title("Runtime Histogram")
sns.distplot(a=dataset1.runtime/360, kde = False, ax = ax[2])
ax[2].set_title("Runtime in Hours Histogram")
sns.distplot(a=np.log1p(dataset1.runtime), kde = False, ax = ax[3])
ax[3].set_title("Log1p transformed Runtime Histogram")
f.tight_layout()

dataset1["runtime_in_hours"] = dataset1.runtime/360
dataset["runtime_in_hours"] = dataset.runtime/360

dataset1["log_runtime"] = np.log1p(dataset1.runtime)
dataset["log_runtime"] = np.log1p(dataset.runtime)

#date preprocessing

from datetime import datetime

# fill possible NA values with the statistical mode
dataset1["release_date"] = dataset1["release_date"].fillna(dataset1["release_date"].mode()[0])
dataset["release_date"] = dataset["release_date"].fillna(dataset["release_date"].mode()[0])


dataset1['temp'] = dataset1.release_date.apply(lambda x: datetime.strptime(x, '%m/%d/%y'))

dataset1["month"] = dataset1.temp.apply(lambda x: x.month)
dataset1["season"] = dataset1["month"]%4
dataset1["year"] = dataset1.temp.apply(lambda x: x.year)
dataset1["day_of_week"] = dataset1.temp.apply(lambda x: x.weekday()+1)
dataset1["week_of_year"] = dataset1.temp.apply(lambda x: x.isocalendar()[1])

dataset1 = dataset1.drop(['temp'], axis=1)


dataset['temp'] = dataset.release_date.apply(lambda x: datetime.strptime(x, '%m/%d/%y'))

dataset["month"] = dataset.temp.apply(lambda x: x.month)
dataset["season"] = dataset["month"]%4
dataset["year"] = dataset.temp.apply(lambda x: x.year)
dataset["day_of_week"] = dataset.temp.apply(lambda x: x.weekday()+1)
dataset["week_of_year"] = dataset.temp.apply(lambda x: x.isocalendar()[1])

dataset = dataset.drop(['temp'], axis=1)



dataset1["day_of_week"] = dataset1["day_of_week"].fillna(dataset1["day_of_week"].mode()[0])
dataset["day_of_week"] = dataset["day_of_week"].fillna(dataset["day_of_week"].mode()[0])

dataset1["year"] = dataset1["year"].fillna(dataset1["year"].mode()[0])
dataset["year"] = dataset["year"].fillna(dataset["year"].mode()[0])

dataset1["month"] = dataset1["month"].fillna(dataset1["month"].mode()[0])
dataset["month"] = dataset["month"].fillna(dataset["month"].mode()[0])

dataset1["week_of_year"] = dataset1["week_of_year"].fillna(dataset1["week_of_year"].mode()[0])
dataset["week_of_year"] = dataset["week_of_year"].fillna(dataset["week_of_year"].mode()[0])

dataset1["season"] = dataset1["season"].fillna(dataset1["season"].mode()[0])
dataset["season"] = dataset["season"].fillna(dataset["season"].mode()[0])

dataset1[["release_date", "month", "year", "day_of_week", "week_of_year", "season"]].head()

#identify the top actor
import re

actors_dict = {}
size_of_actors = len(dataset1) - dataset1.cast.isna().sum()

for element in dataset1[["revenue", "cast"]].values:
    if type(element[1]) == type(str()):
        
        result = re.findall('name\': \'\w+\s*\w*', element[1])
        result = [x.replace("name\': \'", "") for x in result]

        for actor in result:
            if actor not in actors_dict:
                actors_dict[actor] = element[0]
            else:
                actors_dict[actor] += element[0]
                
for actor in actors_dict:
    actors_dict[actor] = actors_dict[actor]/size_of_actors
    


actors_df = pd.DataFrame.from_dict(actors_dict, orient='index', columns=["mean_movies_revenue"])
actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(20).plot.bar()

#feature scaling of top actor

def find_top_actor_from_cast(top_actor, element):
    
    result = []
    if type(element) == type(str()):

        result = re.findall('name\': \'\w+\s*\w*', element)
        result = [x.replace("name\': \'", "") for x in result]
        
    if top_actor in result:
        return 1
    else:
        return 0

for top_actor in actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:
    dataset1["has_top_actor_"+ top_actor] = dataset1.cast.apply(lambda element: find_top_actor_from_cast(top_actor, element))
    dataset["has_top_actor_"+ top_actor] = dataset.cast.apply(lambda element: find_top_actor_from_cast(top_actor, element))


#identify the keywords based on movie revenue
    
import re

keywords_dict = {}
size_of_keywords = len(dataset1) - dataset1.Keywords.isna().sum()

for element in dataset1[["revenue", "Keywords"]].values:
    if type(element[1]) == type(str()):
        
        result = re.findall('name\': \'\w+\s*\w*', element[1])
        result = [x.replace("name\': \'", "") for x in result]

        for key in result:
            if key not in keywords_dict:
                keywords_dict[key] = element[0]
            else:
                keywords_dict[key] += element[0]
                
for key in keywords_dict:
    keywords_dict[key] = keywords_dict[key]/size_of_keywords
    
keywords_df = pd.DataFrame.from_dict(keywords_dict, orient='index', columns=["mean_movies_revenue"])
keywords_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).plot.bar()


# feature scaling identify the keywords based on movie revenue

def find_top_keywords_from_cast(top_keyword, element):
    
    result = []
    if type(element) == type(str()):

        result = re.findall('name\': \'\w+\s*\w*', element)
        result = [x.replace("name\': \'", "") for x in result]
        
    if top_keyword in result:
        return 1
    else:
        return 0

for top_keyword in keywords_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:
    dataset1["has_top_keyword_"+ top_keyword] = dataset1.Keywords.apply(lambda element: find_top_keywords_from_cast(top_keyword, element))
    dataset["has_top_keyword_"+ top_keyword] = dataset.Keywords.apply(lambda element: find_top_keywords_from_cast(top_keyword, element))

# CAST
    


dataset1["num_of_cast"] = dataset1["cast"].str.count("name")
dataset["num_of_cast"] = dataset["cast"].str.count("name")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_cast = dataset1.num_of_cast.fillna(0)
dataset.num_of_cast = dataset.num_of_cast.fillna(0)

sns.boxplot(x=dataset1.num_of_cast, ax = ax[0])
ax[0].set_title("num_of_cast Boxplot")
sns.distplot(a=dataset1.num_of_cast, kde = False, ax = ax[1])
ax[1].set_title("num_of_cast Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_cast), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_cast Histogram")
f.tight_layout()


dataset1["log_num_of_cast"] = np.log1p(dataset1.num_of_cast)
dataset["log_num_of_cast"] = np.log1p(dataset.num_of_cast)


#Male Cast

dataset1["num_of_male_cast"] = dataset1["cast"].str.count("'gender': 2")
dataset["num_of_male_cast"] = dataset["cast"].str.count("'gender': 2")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_male_cast = dataset1.num_of_male_cast.fillna(0)
dataset.num_of_male_cast = dataset.num_of_male_cast.fillna(0)

sns.boxplot(x=dataset1.num_of_male_cast, ax = ax[0])
ax[0].set_title("num_of_male_cast Boxplot")
sns.distplot(a=dataset1.num_of_male_cast, kde = False, ax = ax[1])
ax[1].set_title("num_of_male_cast Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_male_cast), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_male_cast Histogram")
f.tight_layout()


dataset1["log_num_of_male_cast"] = np.log1p(dataset1.num_of_male_cast)
dataset["log_num_of_male_cast"] = np.log1p(dataset.num_of_male_cast)

# Female Cast

dataset1["num_of_female_cast"] = dataset1["cast"].str.count("'gender': 1")
dataset["num_of_female_cast"] = dataset["cast"].str.count("'gender': 1")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_female_cast = dataset1.num_of_female_cast.fillna(0)
dataset.num_of_female_cast = dataset.num_of_female_cast.fillna(0)

sns.boxplot(x=dataset1.num_of_female_cast, ax = ax[0])
ax[0].set_title("num_of_female_cast Boxplot")
sns.distplot(a=dataset1.num_of_female_cast, kde = False, ax = ax[1])
ax[1].set_title("num_of_female_cast Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_female_cast), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_female_cast Histogram")
f.tight_layout()


dataset1["log_num_of_female_cast"] = np.log1p(dataset1.num_of_female_cast)
dataset["log_num_of_female_cast"] = np.log1p(dataset.num_of_female_cast)

# Crew

dataset1["num_of_crew"] = dataset1["crew"].str.count("'job")
dataset["num_of_crew"] = dataset["crew"].str.count("'job")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_crew = dataset1.num_of_crew.fillna(0)
dataset.num_of_crew = dataset.num_of_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_crew, ax = ax[0])
ax[0].set_title("num_of_crew Boxplot")
sns.distplot(a=dataset1.num_of_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_crew Histogram")
f.tight_layout()


dataset1["log_num_of_crew"] = np.log1p(dataset1.num_of_crew)
dataset["log_num_of_crew"] = np.log1p(dataset.num_of_crew)

# counting the number of Male Crew

dataset1["num_of_male_crew"] = dataset1["crew"].str.count("'gender': 2")
dataset["num_of_male_crew"] = dataset["crew"].str.count("'gender': 2")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_male_crew = dataset1.num_of_male_crew.fillna(0)
dataset.num_of_male_crew = dataset.num_of_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_male_crew, ax = ax[0])
ax[0].set_title("num_of_male_crew Boxplot")
sns.distplot(a=dataset1.num_of_male_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_male_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_male_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_male_crew Histogram")
f.tight_layout()


dataset1["log_num_of_male_crew"] = np.log1p(dataset1.num_of_male_crew)
dataset["log_num_of_male_crew"] = np.log1p(dataset.num_of_male_crew)


# counting the number of female crew

dataset1["num_of_female_crew"] = dataset1["crew"].str.count("'gender': 1")
dataset["num_of_female_crew"] = dataset["crew"].str.count("'gender': 1")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_female_crew = dataset1.num_of_female_crew.fillna(0)
dataset.num_of_female_crew = dataset.num_of_female_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_female_crew, ax = ax[0])
ax[0].set_title("num_of_female_crew Boxplot")
sns.distplot(a=dataset1.num_of_female_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_female_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_female_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_female_crew Histogram")
f.tight_layout()


dataset1["log_num_of_female_crew"] = np.log1p(dataset1.num_of_female_crew)
dataset["log_num_of_female_crew"] = np.log1p(dataset.num_of_female_crew)

# identify the top director of the movie
import re

directors_dict = {}
size_of_crew = len(dataset1) - dataset1.crew.isna().sum()

for element in dataset1[["revenue", "crew"]].values:
    if type(element[1]) == type(str()):
        
        result = re.findall('Director\', \'name\': \'\w+\s*\w*', element[1])
        result = [x.replace("Director\', \'name\': \'", "") for x in result]

        for key in result:
            if key not in directors_dict:
                directors_dict[key] = element[0]
            else:
                directors_dict[key] += element[0]
                
for key in directors_dict:
    directors_dict[key] = directors_dict[key]/size_of_crew
    
directors_df = pd.DataFrame.from_dict(directors_dict, orient='index', columns=["mean_movies_revenue"])
directors_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).plot.bar()


#feature scaling and find the top director

def find_top_directors_from_crew(top_director, element):
    
    result = []
    if type(element) == type(str()):

        result = re.findall('Director\', \'name\': \'\w+\s*\w*', element)
        result = [x.replace("Director\', \'name\': \'", "") for x in result]
        
    if top_director in result:
        return 1
    else:
        return 0

for top_director in directors_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:
    dataset1["has_top_director_"+ top_director] = dataset1.crew.apply(lambda element: find_top_directors_from_crew(top_director, element))
    dataset["has_top_director_"+ top_director] = dataset.crew.apply(lambda element: find_top_directors_from_crew(top_director, element))

# identify the top producer of average movie
    
import re

producers_dict = {}
size_of_crew = len(dataset1) - dataset1.crew.isna().sum()

for element in dataset1[["revenue", "crew"]].values:
    if type(element[1]) == type(str()):
        
        result = re.findall('Producer\', \'name\': \'\w+\s*\w*', element[1])
        result = [x.replace("Producer\', \'name\': \'", "") for x in result]

        for key in result:
            if key not in producers_dict:
                producers_dict[key] = element[0]
            else:
                producers_dict[key] += element[0]
                
for key in producers_dict:
    producers_dict[key] = producers_dict[key]/size_of_crew
    
producers_df = pd.DataFrame.from_dict(producers_dict, orient='index', columns=["mean_movies_revenue"])
producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).plot.bar()

#feature scaling of producer

def find_top_producers_from_crew(top_producer, element):
    
    result = []
    if type(element) == type(str()):

        result = re.findall('Director\', \'name\': \'\w+\s*\w*', element)
        result = [x.replace("Director\', \'name\': \'", "") for x in result]
        
    if top_producer in result:
        return 1
    else:
        return 0

for top_producer in producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:
    dataset1["has_top_producer_"+ top_producer] = dataset1.crew.apply(lambda element: find_top_producers_from_crew(top_producer, element))
    dataset["has_top_producer_"+ top_producer] = dataset.crew.apply(lambda element: find_top_producers_from_crew(top_producer, element))

# number of director
    
dataset1["num_of_directors"] = dataset1["crew"].str.count("Directing")
dataset["num_of_directors"] = dataset["crew"].str.count("Directing")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_directors = dataset1.num_of_directors.fillna(0)
dataset.num_of_directors = dataset.num_of_directors.fillna(0)

sns.boxplot(x=dataset1.num_of_directors, ax = ax[0])
ax[0].set_title("num_of_directors Boxplot")
sns.distplot(a=dataset1.num_of_directors, kde = False, ax = ax[1])
ax[1].set_title("num_of_directors Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_directors), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_directors Histogram")
f.tight_layout()


dataset1["log_num_of_directors"] = np.log1p(dataset1.num_of_directors)
dataset["log_num_of_directors"] = np.log1p(dataset.num_of_directors)

# number of producer

dataset1["num_of_producers"] = dataset1["crew"].str.count("Production")
dataset["num_of_producers"] = dataset["crew"].str.count("Production")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_producers = dataset1.num_of_producers.fillna(0)
dataset.num_of_producers = dataset.num_of_producers.fillna(0)

sns.boxplot(x=dataset1.num_of_producers, ax = ax[0])
ax[0].set_title("num_of_producers Boxplot")
sns.distplot(a=dataset1.num_of_producers, kde = False, ax = ax[1])
ax[1].set_title("num_of_producers Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_producers), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_producers Histogram")
f.tight_layout()


dataset1["log_num_of_producers"] = np.log1p(dataset1.num_of_producers)
dataset["log_num_of_producers"] = np.log1p(dataset.num_of_producers)


#number of writer

dataset1["num_of_writers"] = dataset1["crew"].str.count("Writing")
dataset["num_of_writers"] = dataset["crew"].str.count("Writing")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_writers = dataset1.num_of_writers.fillna(0)
dataset.num_of_writers = dataset.num_of_writers.fillna(0)

sns.boxplot(x=dataset1.num_of_writers, ax = ax[0])
ax[0].set_title("num_of_writers Boxplot")
sns.distplot(a=dataset1.num_of_writers, kde = False, ax = ax[1])
ax[1].set_title("num_of_writers Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_writers), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_writers Histogram")
f.tight_layout()


dataset1["log_num_of_writers"] = np.log1p(dataset1.num_of_writers)
dataset["log_num_of_writers"] = np.log1p(dataset.num_of_writers)

# number of editors

dataset1["num_of_editors"] = dataset1["crew"].str.count("Editing")
dataset["num_of_editors"] = dataset["crew"].str.count("Editing")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_editors = dataset1.num_of_editors.fillna(0)
dataset.num_of_editors = dataset.num_of_editors.fillna(0)

sns.boxplot(x=dataset1.num_of_editors, ax = ax[0])
ax[0].set_title("num_of_editors Boxplot")
sns.distplot(a=dataset1.num_of_editors, kde = False, ax = ax[1])
ax[1].set_title("num_of_editors Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_editors), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_editors Histogram")
f.tight_layout()


dataset1["log_num_of_editors"] = np.log1p(dataset1.num_of_editors)
dataset["log_num_of_editors"] = np.log1p(dataset.num_of_editors)

# number of art crew

dataset1["num_of_art_crew"] = dataset1["crew"].str.count("Art")
dataset["num_of_art_crew"] = dataset["crew"].str.count("Art")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_art_crew = dataset1.num_of_art_crew.fillna(0)
dataset.num_of_art_crew = dataset.num_of_art_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_art_crew, ax = ax[0])
ax[0].set_title("num_of_art_crew Boxplot")
sns.distplot(a=dataset1.num_of_art_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_art_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_art_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_art_crew Histogram")
f.tight_layout()


dataset1["log_num_of_art_crew"] = np.log1p(dataset1.num_of_art_crew)
dataset["log_num_of_art_crew"] = np.log1p(dataset.num_of_art_crew)

#number of sound crew

dataset1["num_of_sound_crew"] = dataset1["crew"].str.count("Sound")
dataset["num_of_sound_crew"] = dataset["crew"].str.count("Sound")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_sound_crew = dataset1.num_of_sound_crew.fillna(0)
dataset.num_of_sound_crew = dataset.num_of_sound_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_sound_crew, ax = ax[0])
ax[0].set_title("num_of_sound_crew Boxplot")
sns.distplot(a=dataset1.num_of_sound_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_sound_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_sound_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_sound_crew Histogram")
f.tight_layout()


dataset1["log_num_of_sound_crew"] = np.log1p(dataset1.num_of_sound_crew)
dataset["log_num_of_sound_crew"] = np.log1p(dataset.num_of_sound_crew)

#number of crew and makeup team

dataset1["num_of_costume_crew"] = dataset1["crew"].str.count("Costume & Make-Up")
dataset["num_of_costume_crew"] = dataset["crew"].str.count("Costume & Make-Up")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_costume_crew = dataset1.num_of_costume_crew.fillna(0)
dataset.num_of_costume_crew = dataset.num_of_costume_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_costume_crew, ax = ax[0])
ax[0].set_title("num_of_costume_crew Boxplot")
sns.distplot(a=dataset1.num_of_costume_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_costume_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_costume_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_costume_crew Histogram")
f.tight_layout()


dataset1["log_num_of_costume_crew"] = np.log1p(dataset1.num_of_costume_crew)
dataset["log_num_of_costume_crew"] = np.log1p(dataset.num_of_costume_crew)


#number of camera crew

dataset1["num_of_camera_crew"] = dataset1["crew"].str.count("\'department\': \'Camera\'")
dataset["num_of_camera_crew"] = dataset["crew"].str.count("\'department\': \'Camera\'")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_camera_crew = dataset1.num_of_camera_crew.fillna(0)
dataset.num_of_camera_crew = dataset.num_of_camera_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_camera_crew, ax = ax[0])
ax[0].set_title("num_of_camera_crew Boxplot")
sns.distplot(a=dataset1.num_of_camera_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_camera_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_camera_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_camera_crew Histogram")
f.tight_layout()


dataset1["log_num_of_camera_crew"] = np.log1p(dataset1.num_of_camera_crew)
dataset["log_num_of_camera_crew"] = np.log1p(dataset.num_of_camera_crew)


# number of visual effect crew

dataset1["num_of_visual_effects_crew"] = dataset1["crew"].str.count("\'department\': \'Visual Effects\'")
dataset["num_of_visual_effects_crew"] = dataset["crew"].str.count("\'department\': \'Visual Effects\'")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_visual_effects_crew = dataset1.num_of_visual_effects_crew.fillna(0)
dataset.num_of_visual_effects_crew = dataset.num_of_visual_effects_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_visual_effects_crew, ax = ax[0])
ax[0].set_title("num_of_visual_effects_crew Boxplot")
sns.distplot(a=dataset1.num_of_visual_effects_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_visual_effects_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_visual_effects_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_visual_effects_crew Histogram")
f.tight_layout()


dataset1["log_num_of_visual_effects_crew"] = np.log1p(dataset1.num_of_visual_effects_crew)
dataset["log_num_of_visual_effects_crew"] = np.log1p(dataset.num_of_visual_effects_crew)


#number of lighting crew

dataset1["num_of_lighting_crew"] = dataset1["crew"].str.count("\'department\': \'Lighting\'")
dataset["num_of_lighting_crew"] = dataset["crew"].str.count("\'department\': \'Lighting\'")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_lighting_crew = dataset1.num_of_lighting_crew.fillna(0)
dataset.num_of_lighting_crew = dataset.num_of_lighting_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_lighting_crew, ax = ax[0])
ax[0].set_title("num_of_lighting_crew Boxplot")
sns.distplot(a=dataset1.num_of_lighting_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_lighting_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_lighting_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_lighting_crew Histogram")
f.tight_layout()


dataset1["log_num_of_lighting_crew"] = np.log1p(dataset1.num_of_lighting_crew)
dataset["log_num_of_lighting_crew"] = np.log1p(dataset.num_of_lighting_crew)

# number of other crew

dataset1["num_of_other_crew"] = dataset1["crew"].str.count("\'department\': \'Crew\'")
dataset["num_of_other_crew"] = dataset["crew"].str.count("\'department\': \'Crew\'")

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_other_crew = dataset1.num_of_other_crew.fillna(0)
dataset.num_of_other_crew = dataset.num_of_other_crew.fillna(0)

sns.boxplot(x=dataset1.num_of_other_crew, ax = ax[0])
ax[0].set_title("num_of_other_crew Boxplot")
sns.distplot(a=dataset1.num_of_other_crew, kde = False, ax = ax[1])
ax[1].set_title("num_of_other_crew Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_other_crew), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_other_crew Histogram")
f.tight_layout()


dataset1["log_num_of_other_crew"] = np.log1p(dataset1.num_of_other_crew)
dataset["log_num_of_other_crew"] = np.log1p(dataset.num_of_other_crew)


#production countries

dataset1["num_of_production_countries"] = dataset1.production_countries_processed.apply(len)
dataset["num_of_production_countries"] = dataset.production_countries_processed.apply(len)

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_production_countries = dataset1.num_of_production_countries.fillna(0)
dataset.num_of_production_countries = dataset.num_of_production_countries.fillna(0)

sns.boxplot(x=dataset1.num_of_production_countries, ax = ax[0])
ax[0].set_title("num_of_production_countries Boxplot")
sns.distplot(a=dataset1.num_of_production_countries, kde = False, ax = ax[1])
ax[1].set_title("num_of_production_countries Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_production_countries), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_production_countries Histogram")
f.tight_layout()


dataset1["log_num_of_production_countries"] = np.log1p(dataset1.num_of_production_countries)
dataset["log_num_of_production_countries"] = np.log1p(dataset.num_of_production_countries)

#number of genres in a movie

dataset1["num_of_genres"] = dataset1.genres_processed.apply(len)
dataset["num_of_genres"] = dataset.genres_processed.apply(len)

f, ax = plt.subplots(3, figsize=(12,7))

dataset1.num_of_genres = dataset1.num_of_genres.fillna(0)
dataset.num_of_genres = dataset.num_of_genres.fillna(0)

sns.boxplot(x=dataset1.num_of_genres, ax = ax[0])
ax[0].set_title("num_of_genres Boxplot")
sns.distplot(a=dataset1.num_of_genres, kde = False, ax = ax[1])
ax[1].set_title("num_of_genres Histogram")
sns.distplot(a=np.log1p(dataset1.num_of_genres), kde = False, ax = ax[2])
ax[2].set_title("Log1p transformed num_of_genres Histogram")
f.tight_layout()


dataset1["log_num_of_genres"] = np.log1p(dataset1.num_of_genres)
dataset["log_num_of_genres"] = np.log1p(dataset.num_of_genres)


# Project is still not completed .
# Only the data preprocessing is done ..
