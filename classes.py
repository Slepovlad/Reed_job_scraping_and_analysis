import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re

class ReedJobScraper:
    def __init__(self, job_list):
        self.job_list = job_list

    def __get_jobs(self, job_title: str, page_number: int) -> pd.DataFrame:
        """
        Scrapes job information from the Reed job search website based on a given job title and page number.

        Args:
            job_title (str): The job title to search for. Spaces in the job title will be replaced with hyphens.
            page_number (int): The page number of the search results to retrieve.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing job information, including the job title, data-gtm attribute,
            data-id attribute, job link, and page number.
        """
        list_dict = []
        job_title = job_title.replace(' ','-').casefold()
        page_number = int(page_number)
        if page_number == 1:
            url = f"https://www.reed.co.uk/jobs/{job_title}-jobs"
        else:
            url = f"https://www.reed.co.uk/jobs/{job_title}-jobs?pageno={page_number}"

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        jobs = soup.find_all('h2', class_ = 'job-result-heading__title')
        for job in jobs:
            info = job.find('a', href=True)
            dict_ = {'title' : info['title'],
                    'data_gtm' : info['data-gtm'],
                     'data_id' : info['data-id'],
                     'link' : info['href'],
                     'page' : page_number}
            list_dict.append(dict_)

        df_result = pd.DataFrame.from_dict(list_dict)

        return df_result

    def __collect_from_all_pages(self, job_title: str) -> pd.DataFrame:
        """
        Collects job information from all pages of search results for a given job title on the Reed job search website.

        Args:
            job_title (str): The job title to search for. Spaces in the job title will be replaced with hyphens.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing job information, including the job title, data-gtm attribute,
            data-id attribute, job link, and page number for all pages of search results.
        """
        list_df = []
        i = 1 
        while True:
            df_temp = self.__get_jobs(job_title, i)
            if len(df_temp) == 0:
                break
            else:
                list_df.append(df_temp)
            time.sleep(1)
            print(f'{job_title} page {i} collected')
            i+=1

        return pd.concat(list_df)

    def collect_all(self) -> dict:
        """
        Collects job postings data for all job titles in the self.job_list attribute.
            
        Returns:
            dict: A dictionary of job titles as keys and their corresponding data as pandas dataframes.
        """
        dict_of_dfs = {}
        for job_title in self.job_list:
            dict_of_dfs[job_title] = self.__collect_from_all_pages(job_title)

        self.dict_of_dfs = dict_of_dfs

        return dict_of_dfs

    def concat_and_drop_duplicates(self) -> pd.DataFrame:

        """
        Concatenates all dataframes in self.dict_of_dfs and removes any duplicate rows based on the 'data_id' column.
        The resulting concatenated dataframe is saved as self.df_full attribute.

        Returns:
            pd.DataFrame: The concatenated and dataframe without duplicates.
        """

        df_full = pd.concat(self.dict_of_dfs.values())
        df_full = df_full.drop_duplicates(subset=['data_id'])
        df_full = df_full.reset_index(drop = True)
        self.df_full = df_full

        return df_full

    
    def save_all(self):
        """
        Saves all dataframes in self.dict_of_dfs as separate csv files with dates and saves the self.df_full as a csv file.
        """

        for name,df in self.dict_of_dfs.items():
            df.to_csv(f'{name}_list_{datetime.datetime.today().strftime("%Y_%m_%d")}.csv')

        self.df_full.to_csv(f'all_jobs_list_{datetime.datetime.today().strftime("%Y_%m_%d")}.csv')
    
    def __select_text(self, soup, selector) -> str:
        """
        Extracts the text content from the HTML element that matches the given CSS selector. If not possible returns None.

        Args:
            soup (BeautifulSoup): A BeautifulSoup object.
            selector (str): A CSS selector used to find the target element.

        Returns:
            str: The text content of the target HTML element if it exists, otherwise None.
        """

        text = soup.select_one(selector)
        if soup.select_one(selector) == None:
            return None
        else:
            return text.text
    
    def __get_job_info(self,link) -> dict:
        """
        Extracts job information from a given job listing URL.

        Args:
            link (str): A relative URL representing a job listing on reed.co.uk.

        Returns:
            dict: A dictionary containing the extracted job information, with keys 'link', 'title', 'company_name', 'salary',
            'permanent', 'full_time', 'date_posted', 'region', 'city', 'description_clean', 'description_html', and 'skills'.
        """
    
        response = requests.get('https://www.reed.co.uk'+link)
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = self.__select_text(soup,'h1')
        company_name = self.__select_text(soup,'span [itemprop="name"]')
        try:
            salary = soup.select_one('[itemprop="currency"]').next_element.next_element.text
        except:
            salary = None
        
        if soup.select_one('span [href="/jobs/permanent"]') != None:
            permanent = True
        else:
            permanent = False
            
        if soup.select_one(' span [href="/jobs/full-time"]') != None:
            full_time = True
        else:
            full_time = False
        
        try:
            date_posted = soup.select_one(' [itemprop = "datePosted"]')['content']
        except:
            date_posted = None
            
        region = self.__select_text(soup, ' [data-qa="regionMobileLbl"]')
        if region == None:
            region = self.__select_text(soup, ' [data-qa="regionLbl"]') #sponsored links have different structure
        
        city = self.__select_text(soup,' [data-qa="localityMobileLbl"]')
        if city == None:
            city = self.__select_text(soup,' [data-qa="localityLbl"]')
        
        description_clean = self.__select_text(soup,'[itemprop="description"]')
        description_html = str(soup.select_one('[itemprop="description"]'))
        
        skills = ''
        for skill in soup.select('.skills li'):
            skills+=(skill.text+',')
        
        return {'link':link,
                'title':title,
                'company_name':company_name,
                'salary':salary,
                'permanent':permanent,
                'full_time':full_time,
                'date_posted':date_posted,
                'region':region,
                'city':city,
                'description_clean':description_clean,
                'description_html':description_html,
                'skills':skills}
    
    def collect_all_descriptions(self,sleep_time = 0.1) -> pd.DataFrame:
        """
        Collects job information for all links in `self.df_full`.
        
        Args:
            sleep_time (float, optional): Number of seconds to wait before each request to the website. 
                                        Defaults to 0.1.
                                        
        Returns:
            pd.DataFrame: DataFrame containing job information for all links in `self.df_full`, including 
                        job descriptions.
        """

        list_of_jobs_info = [] 
        i = 0
        for link in self.df_full.link:
            list_of_jobs_info.append(self.__get_job_info(link))
            time.sleep(sleep_time)
            i+=1
            if i%10 == 0:
                print(f'parsed {i} jobs')
        
        df_full_with_descriptions = pd.DataFrame.from_dict(list_of_jobs_info)
        #this part is needed for data to be saved 
        for col in df_full_with_descriptions.columns:
            if df_full_with_descriptions[col].dtype==object:
                df_full_with_descriptions[col]=df_full_with_descriptions[col].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))

        df_full_with_descriptions.to_csv(f'all_jobs_list_with_descriptions_{datetime.datetime.today().strftime("%Y_%m_%d")}.csv')
        self.df_full_with_descriptions = df_full_with_descriptions
        
        return df_full_with_descriptions
   













class JobsCleaning:

    def __init__(self, initial_df):
        self.initial_df = initial_df
        self.df_clean = None
        self.cleaning_used = {'lower_case':'',
                              'salary':'',
                              'region_to_city':'',
                              'jobs_to_consider': '',
                              'only_considered_jobs': False,
                              'position_list': '',
                              }

    def __salary_cleaning(self, salary) -> tuple:
        """
        Cleans the salary string and returns its parts.

        Args:
            salary (str): The salary string.

        Returns:
            tuple: A tuple containing the minimum and maximum salary, the time period, 
            and the currency symbol extracted from the salary string.

        """

        salary = salary.casefold()
        salary = salary.replace(',','')
        salary = salary.replace('usd','')
        
        period = None
        salary_min = None
        salary_max = None
        currency = None 
        
        for c in ['aud$','£','€','$']:
            if salary.find(c) != -1:
                currency = c
                salary = salary.replace(c,'')
                break
                
        # if salary is filled always has 'per' word
        per_index = salary.find('per') 
        if per_index == -1:
            return salary_min, salary_max, period, currency
        else: 
            period = salary[per_index:].split()[1]
            salary_split = salary[:per_index].split()
            
            if len(salary_split) == 1:
                salary_min = salary_split[0]
                salary_max = salary_split[0]
                
            else:
                salary_min = salary_split[0]
                salary_max = salary_split[-1]

        return salary_min, salary_max, period, currency

    def __df_to_use(self) -> pd.DataFrame:
        """
        Returns the dataframe to use for cleaning based on the availability of a cleaned dataframe.

        Returns:
            pd.DataFrame: The dataframe to use for cleaning.
        """

        if self.df_clean is None:
            df_to_use = self.initial_df.copy()
        else:
            df_to_use = self.df_clean.copy()

        return df_to_use

    def add_salary_columns(self) -> pd.DataFrame:
        """
        Cleans the salary column of the dataframe and creates columns for the minimum, maximum, mean salaries,
        the time period, and the currency.

        Returns:
            pd.DataFrame: The cleaned dataframe with the new columns for the salary details.
        """

        df_to_use = self.__df_to_use()
        df_temp = df_to_use.salary.apply(self.__salary_cleaning)
        df_temp = pd.DataFrame(df_temp.to_list(),index=df_to_use.index, columns = ['salary_from', 'salary_to', 'time-period','currency'])
        df_temp.salary_from = df_temp.salary_from.fillna(0).astype('float')
        df_temp.salary_to = df_temp.salary_to.fillna(0).astype('float')

        df_temp['mean_salary'] = (df_temp.salary_from + df_temp.salary_to)/2

        df_clean = pd.concat([df_to_use, df_temp], axis = 1)

        self.df_clean = df_clean
        self.cleaning_used['salary'] = 'salary cleaned'

        return df_clean
    
    def make_lower_case(self, list_of_columns = ['title', 'description_clean', 'skills']) -> pd.DataFrame:
        """
        Converts the text in specified columns of the dataframe to lowercase.

        Args:
            list_of_columns (list): A list of column names to convert to lowercase. Default is ['title', 'description_clean', 'skills'].

        Returns:
            pd.DataFrame: The cleaned dataframe with the specified columns converted to lowercase.
        """

        df_to_use = self.__df_to_use()
        casefolding = lambda x: x.casefold()
        for column in list_of_columns:
            df_to_use[column] = df_to_use[column].fillna('no value').apply(casefolding)
            df_to_use[column] = df_to_use[column].replace('no value',np.nan)
            self.cleaning_used['lower_case'] = self.cleaning_used.get('lower_case') + ', ' + column
        self.df_clean = df_to_use

        return df_to_use
    
    def replace_city(self, cities = []) -> pd.DataFrame:
        """
        Replaces the region value of each row with the corresponding city name.

        Args:
            cities (list): A list of city names to replace the region value with.

        Returns:
            pd.DataFrame: The cleaned dataframe with the updated city values.
        
        Example:
            Sometimes London mentioned in region and not in city and in city part of London mentioned such as "Canary Wharf".
            If we want to replace all the occurrences of the string 'London' in the 'region' column with the string 'London', we can use the following code:
        
        >>> cleaner = JobListingCleaner(initial_df)
        >>> cleaner.replace_city(cities=['London'])
        
        """

        df_to_use = self.__df_to_use()
        for city in cities:
            df_to_use.loc[df_to_use['region'] == city, 'city'] = city
            self.cleaning_used['region_to_city'] = self.cleaning_used.get('region_to_city') + ', ' + city
        self.df_clean = df_to_use

        return df_to_use
    
    def __to_consider(self, title, jobs_to_look_for):
        """
        Checks if a job title contains any of the given job keywords.

        Args:
            title (string): A job title to check for keywords.
            jobs_to_look_for (list): A list of job keywords to look for in the job title.

        Returns:
            If any keyword in the job keyword list is found in the job title, returns the keyword. Otherwise, returns None.
        """

        for job in jobs_to_look_for:
            i = 0
            jop_parts = job.casefold().split()
            for job_part in jop_parts:
                if title.casefold().find(job_part) == -1:
                    break
                else:
                    i+=1
            if i==len(jop_parts):
                return job

        return None
    
    def jobs_to_look_for(self, jobs_to_look_for) -> pd.DataFrame:
        """
        Identifies job titles in the 'title' column of the DataFrame that match a given list of job titles.
        
        Args:
            jobs_to_look_for: A list of job titles to search for in the 'title' column of the DataFrame.
        
        Returns:
             pd.DataFrame: A DataFrame with an additional column 'jobs_to_consider'.
        """

        df_to_use = self.__df_to_use()
        to_consider_apply = lambda x :self.__to_consider(x, jobs_to_look_for)
        df_to_use['jobs_to_consider'] = df_to_use.title.apply(to_consider_apply)
        self.cleaning_used['jobs_to_consider'] = jobs_to_look_for
        print(df_to_use.jobs_to_consider.value_counts()[:10])
        self.df_clean = df_to_use

        return df_to_use

    def extract_position(self, position_list = ['senior','lead','junior','principal','graduate']):
        """
        Identifies position from job titles in the 'title' column of the DataFrame that match a given list of job positions in terms of experience.
        
        Args:
            position_list: A list of job titles to search for in the 'title' column of the DataFrame. 
                Default position_list = ['senior','lead','junior','principal','graduate'].
        
        Returns:
             pd.DataFrame: A DataFrame with an additional column 'position'.
        """
        
        df_to_use = self.__df_to_use()
        to_consider_apply = lambda x :self.__to_consider(x, position_list)
        df_to_use['position'] = df_to_use.title.apply(to_consider_apply)
        df_to_use['position'].fillna('No position in title')
        self.cleaning_used['position_list'] = position_list
        self.df_clean = df_to_use

        return df_to_use

    def only_leave_considered(self):
        """
        Drops all rows from the DataFrame where the 'jobs_to_consider' column has missing values.
        """

        self.df_clean = self.df_clean.dropna(subset = ['jobs_to_consider'])
        self.cleaning_used['only_considered_jobs'] = True

    def save_clean_version(self):
        """
        Saves the cleaned version of the DataFrame to a CSV file with the current date in the filename.
        Prints the list of cleaning steps that were applied to the DataFrame.
        """

        self.df_clean.to_csv(f'cleaned_jobs_with_descriptions_{datetime.datetime.today().strftime("%Y_%m_%d")}.csv', index = False)
        print(self.cleaning_used)
















class JobsVisualisation:

    def __init__(self, initial_df) -> None:

        self.initial_df = initial_df
        self.extacted_skills_df = None 
        self.extracted_skills_figs = {}

    def create_salary_box_plots(self, min_salary: int = 12000, max_salary: int = 160000):
        """Creates box plots of salaries for selected job types. Also saves dataframe used to show this figure and figure itself.

        Args:
            min_salary (int, optional): The minimum salary for which to show box plots. Defaults to 12000.
            max_salary (int, optional): The maximum salary for which to show box plots. Defaults to 160000.

        """

        df_box_plots = self.initial_df.copy()

        df_box_plots = df_box_plots[['jobs_to_consider','mean_salary','permanent','full_time','time-period','currency']]
        df_box_plots = df_box_plots[df_box_plots.currency == '£'] # only leave if currency is equal to £
        df_box_plots = df_box_plots[df_box_plots['time-period'] == 'annum'] #only leave annum salaries
        df_box_plots = df_box_plots[df_box_plots.full_time == True] #only full time jobs
        #restrictions to remove outliers (potential typos when salary was filled)
        df_box_plots = df_box_plots[(df_box_plots.mean_salary<max_salary) & (df_box_plots.mean_salary>min_salary)] 

        fig = px.box(df_box_plots[['jobs_to_consider','mean_salary']],x = "jobs_to_consider", y="mean_salary",
        labels=dict(jobs_to_consider="Job", mean_salary="Salary (£)"))
        self.box_plot_df = df_box_plots
        self.box_plot_fig = fig
        fig.show()
    
    def plot_map(self, min_salary: int = 12000, max_salary: int = 160000):
        """Plot a map of the UK with markers for job vacancies. Use data from https://simplemaps.com/data/gb-cities# to map cities. 
        Also saves dataframe used to show this map and map itself.
        High number of vacancies -> big bubbles 
        High medium salary -> lighter bubbles

        Args:
            min_salary (int): The minimum salary for which to show job vacancies.
            max_salary (int): The maximum salary for which to show job vacancies.
        """
        #https://simplemaps.com/data/gb-cities#
        #link with gb-cities  
        df_cities = pd.read_csv(r'gb.csv')  
        self.df_cities = df_cities

        df_map = self.initial_df.copy()
        df_map = df_map[['jobs_to_consider','city','region','mean_salary','permanent','full_time','time-period','currency']]
        df_map = df_map[df_map.city != 'None']
        df_cities = df_cities.drop(columns = ['country','capital'])

        df_map = df_map.merge(df_cities, on = 'city', how = 'outer')
        df_map = df_map[df_map.permanent.notna()]
        df_map = df_map[(df_map.mean_salary<max_salary) & (df_map.mean_salary>min_salary)] 
        df_map = df_map[df_map.lat.notna()]
        print('total number of jobs on map:')
        print(df_map.jobs_to_consider.value_counts())

        df_temp = df_map[['city','mean_salary','lat','lng']].groupby('city').mean()
        df_map = pd.concat([df_temp,pd.DataFrame(df_map.city.value_counts())], axis = 1)

        df_map['size'] = df_map.city * 100
        df_map['size'] = df_map['size'].apply(np.sqrt)

        #df_map.mean_salary = df_temp['mean_salary'].astype('int')
        #df_map['size'] = df_temp['size'].astype('int')
        df_map = df_map.rename(columns = {'city': 'number of vacancies', 'mean_salary': 'median salary'})

        fig = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lng",
            color = "median salary",
            hover_name=df_map.index,
            hover_data =['number of vacancies'],
            size="size",
            size_max = 40
        ).update_layout(mapbox={"style": "carto-positron", "zoom": 6}, margin={"t":0,"b":0,"l":0,"r":0})
        self.map_df = df_map
        self.map_fig = fig
        fig.show()
    
    def plot_skills_bar_chart(self, display = 15, normalisation = False):
        """
        Plots a bar chart of the most common skills for each job. Also saves dataframe used to show this figure and figure itself.

        Args:
            display (int): The number of skills to display for each job. Default is 15.
            normalisation (bool): Whether or not to normalize the skill count by the number of jobs. Default is False.
        """

        df_skills = self.initial_df.copy()

        df_skills = df_skills[df_skills.skills.notna()]
        normaliser = df_skills.jobs_to_consider.value_counts().to_dict() #divide skills by number of jobs to get percentage
        df_skills = df_skills[['jobs_to_consider','skills']].groupby('jobs_to_consider').sum()


        skills_cleaning = lambda x: x.casefold().replace('"',"")
        skills_into_list = lambda x: re.split('[,.|;/]',x)

        df_skills = df_skills.applymap(skills_cleaning)
        df_skills = df_skills.applymap(skills_into_list)

        skills_dict = {}

        for i in df_skills.index: # each index equal to job which is going to be visualised
            skills_dict[i] = pd.Series(df_skills.loc[i].values[0]).value_counts()
        
        fig, ax = plt.subplots(nrows=len(skills_dict), ncols=1, figsize=(18, 15))
        plt.subplots_adjust(hspace=0.7)
        plt.rc('axes', labelsize=100) 
        
        i = 0 
        for job, skills in skills_dict.items():
            
            if normalisation == False:
                data = skills.values[:display]
                labels = skills.index[:display]
            else:
                data = skills.values[:display]/normaliser[job]
                labels = skills.index[:display]
            
            ax[i].bar(labels,data)
            ax[i].set_title(job+' skills', fontdict = {'fontsize': 15})
            ax[i].tick_params(axis="x", direction="in", pad=10, rotation = 45, labelsize = 11 )
            i+=1

        self.skills_df = df_skills
        self.bar_plt = fig
        plt.show()

    def __search_key_word(self, text:str, word:str):
        """
        Helper method to search for a specific word in a given text.
        Args:
            text (str): A string of text to search for a keyword.
            word (str): A string of the keyword to search for in the text.

        Returns:
            int: Returns 1 if the keyword is found in the text, and 0 otherwise.
        """

        if text.find(word) != -1:
            return 1
        else: 
            return 0

    def create_columns_based_on_words(self, search_list:list):
        """
        Create new columns in the initial dataframe based on the presence of specific search words.
        And saves new dataframe with new columns.

        Args:
            search_list (list): List of strings representing search words.

        Examples:
            >>> job_search = JobSearch()
            >>> job_search.create_columns_based_on_words(['python', 'sql'])
            >>> print(job_search.extacted_skills_df.head())
        """
        
        if self.extacted_skills_df is None:
            df_key_technologies = self.initial_df.copy()
            df_key_technologies = df_key_technologies[['jobs_to_consider','description_clean', 'skills', 'mean_salary']]
            df_key_technologies = df_key_technologies.fillna('no')
            df_key_technologies['descr_and_skills'] = df_key_technologies['description_clean']+' '+df_key_technologies['skills']
            casefolding = lambda x: x.casefold()
            df_key_technologies['descr_and_skills'] = df_key_technologies['descr_and_skills'].apply(casefolding)
        else:
            df_key_technologies = self.extacted_skills_df.copy()
        
        for search_word in search_list:
            search_key_word_apply = lambda x: self.__search_key_word(x,search_word)
            df_key_technologies[search_word] = df_key_technologies['descr_and_skills'].apply(search_key_word_apply)
        
        self.extacted_skills_df = df_key_technologies
        self.extacted_skills_normalisation = df_key_technologies.jobs_to_consider.value_counts().to_dict()

    
    def plot_extacted_skills(self, list_of_tech:list, normalisation:bool = False, name:str = 'name'):
        """
        Plot the extracted skills across all jobs and for each job category separately. 
        If normalisation is set to True normalise by the number of jobs.
        Also save figure in `self.extracted_skills_figs` dictionary under name. 

         Args:
            list_of_tech (list): The list of technologies to plot.
            normalisation (bool, optional): Set to True to normalize the results. Defaults to False.
            name (str, optional): The name of the plot. Defaults to 'name'.

        Returns:
        None: The function plots the extracted skills.

        Raises:
        AssertionError: If normalisation is set to True, but the normalization dictionary is not dictionary or its length is not equal to the number of jobs.

        """

        if self.extacted_skills_df is None:
            print('Use create_columns_based_on_words at least ones')
            return None
        else:
            df_to_use = self.extacted_skills_df

        columns = df_to_use.jobs_to_consider.unique()
        n_columns = len(columns)
        
        if normalisation == True:
            normaliser = self.extacted_skills_normalisation
            assert (type(normaliser) == dict)
            assert (len(normaliser) == n_columns)
        
        fig = plt.figure(constrained_layout=True, figsize=(18, 10))
        G = fig.add_gridspec(2, n_columns)
        ax1 = fig.add_subplot(G[0,:])
        ax1.set_title('Across all jobs')
        
        if normalisation == True:
            normalise_all_value = sum(normaliser.values())
        else:
            normalise_all_value = 1
            
        data_all = df_to_use[list_of_tech].sum()
        data_all = data_all.sort_values(ascending = False)
        values_all = data_all.values/normalise_all_value
        labels_all = data_all.index
        ax1.bar(labels_all,values_all)
        
        list_of_axes = []
        for i in range(n_columns):
            list_of_axes.append(fig.add_subplot(G[1,i]))
            
            data_job_specific = df_to_use[df_to_use.jobs_to_consider == columns[i]][list_of_tech].sum()
            data_job_specific = data_job_specific.sort_values(ascending = False)
            
            if normalisation == True:
                normalise_specific_value = normaliser[columns[i]]
            else:
                normalise_specific_value = 1
                
            values_job_specific = data_job_specific.values/normalise_specific_value
            labels_job_specific = data_job_specific.index
            
            ax = list_of_axes[i]
            ax.bar(labels_job_specific,values_job_specific)
            ax.set_title(columns[i])

        self.extracted_skills_figs[name] = fig
        plt.show()
    
    def create_company_pie_chart(self, number_to_show = 10):

        df_pie_chart = self.initial_df.copy()
        df_pie_chart = df_pie_chart.company_name.value_counts()

        if len(df_pie_chart) > number_to_show:
            df_pie_chart = pd.concat([df_pie_chart[:number_to_show],pd.Series({'Other':df_pie_chart[number_to_show:].sum()})])
        
        fig = px.pie(values=df_pie_chart.values, names = df_pie_chart.keys(), title='Who is hiring?', hole=.3)

        self.pie_chart_fig = fig
        fig.show()


        

    
