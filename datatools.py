import re
import pandas as pd
import geopandas as gpd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def df_col(columns,max_length=15):

    try:
        columns= [string.lower() for string in columns]
        for string in columns:
            index = columns.index(string)
            pattern = r'\s\([^)]*\)'
            string = re.sub(pattern,'',string)
            columns[index] = string
            if len(string) > max_length:
                index = columns.index(string)
                pattern1 = r'\b\w'
                initials = re.findall(pattern1,string)
                initials = [letter.lower() for letter in initials]
                string = ''.join(initials)
                columns[index] = string
        pattern2 = r'\s'
        replacement = '_'
        columns = [re.sub(pattern2,replacement,col) for col in columns]

        return columns
    
    except IndexError as e:
        print(f'Index Error: {e}')
        return None
    
    except re.error as e:
        print(f'Regular Expression Error: {e}')
        return None


def extract_cordinates(cordinates):

    if not isinstance(cordinates,list):
        raise ValueError('Input must be a List of cordinate strings!')
        
    try:
        latitude = list()
        longitude = list()
        pattern = r'[+-]?\d*\.\d+|\d+'
        for i,x in enumerate(cordinates):
            match = re.findall(pattern,x)
            latitude.append(match[0])
            longitude.append(match[1])
        return latitude,longitude
    
    except re.error as e:
        print(f'Regular Expression Error: {e}')
    
    except ValueError as e:
        print(f'Value Error: {e}')
    

def load_csv(file_path):

    try:
        df = pd.read_csv(file_path)
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None



def load_xlsx(file_path):

    try:
        df = pd.read_excel(file_path,sheet_name=None)
        return df
    
    except FileNotFoundError:
        print(f'Error: File not found at {file_path}')
        return None
    
    
def trendline(df,x,y,deg=1,extend=None,precision=1,linestyle='--',color='r',marker=None,label='Trendline'):
    try:
        coef=np.polyfit(df[x],df[y],deg)
        trend=np.poly1d(coef)
        if df[x].dtype==float:
            x_val=df[x]
        if extend==None:
            x_val=range(min(df[x]),max(df[x]),precision)
        if extend<min(df[x]):
            x_val=range(extend,max(df[x]),precision)
        if (extend>max(df[x])):
            x_val=range(min(df[x]),extend,precision)
        y_val=trend(x_val)
        plt.plot(x_val, y_val, color=color,linestyle=linestyle,label=label,marker=marker)
    
    
    except ValueError as e:
        print(f'Value error: {e}')
        return None
    
    except TypeError as e:
        print(f'Type error: {e}')
        return None
    
    
    
def radar_chart(df,categories,values,title=None,fill=True,color='blue'):
    try:
        if not isinstance(categories,list):
            categories = list(df[categories])

        if not isinstance(values,list):
            values = list(df[values])

        values.append(values[0])
        num_categories = len(categories)

        angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots()
        
        if fill==True:
            plt.fill(angles, values, color, alpha=0.1)
        ax.plot(angles, values, color, linewidth=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        ax.fill(angles, values, 'white', alpha=0.1)

        plt.title(title)
        plt.show()
        
    except ValueError as e:
        print(f'Value error: {e}')
        return None
    
    except TypeError as e:
        print(f'Type error: {e}')
        return None


def geo_plot(df,latitude,longitude,file_path,title=None,size=(10,10),marker_size=10):

    if not (isinstance(latitude,pd.Series) or isinstance(longitude,pd.Series)):
        raise TypeError('Latitude and Longitude must be entered as Pandas Series.\nPlease refer to the documentation on how to create Pandas Series for latitude and longitude data.')
        
    try:
        map_bg = gpd.read_file(file_path)
        gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(latitude,longitude),crs='EPSG:4326')
        ax=map_bg.plot(color='lightgray', edgecolor='black',legend=False,figsize=size)
        gdf.plot(marker='o', color='blue',markersize=10,alpha=0.7,ax=ax)
        plt.title(title)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.show()
        
    except FileNotFoundError:
        print(f'Error: File not found at {file_path}')
        return None
    
    except ValueError as e:
        print(f'Value error: {e}')
        return None
    
    except TypeError as e:
        print(f'Type error: {e}')
        return None
    
    
    
def corr_mat(df):

    try:   
        mat = df.corr()
        sns.heatmap(data=mat, annot=True, fmt='.2f')
        plt.title('Correlation Matrix of Numeric Columns')
        plt.show()
        
    except ValueError as e:
        print(f'Value Error: {e}')
        
    except TypeError as e:
        print(f'Type Error: {e}')


        
def lr_curve(epoch_count,loss_count,test_count):
    
    try:
        plt.plot(epoch_count,loss_count,c='blue',label="Test Loss")
        plt.plot(epoch_count,test_count,c='red',label="Test Loss")
        plt.title("Learning Curve",fontsize=20)
        plt.xlabel("Epoch",fontsize=12)
        plt.ylabel("Loss",fontsize=12)
        plt.grid(linestyle='-.')
        plt.legend()
        plt.minorticks_on()
        plt.show()
        
        
    except ValueError as e:
        print(f'Value error: {e}')
        return None
    
    except TypeError as e:
        print(f'Type error: {e}')
        return None

    
    
df_col.__doc__=    """
    Clean and format a list of DataFrame column names.

    This function takes a list of column names and performs the following operations:

    1. Convert all column names to lowercase.
    2. Remove parentheses and their contents (e.g., '(Year)' becomes 'Year').
    3. Truncate column names to the specified `max_length` by using initials (e.g., 'ColumnExample' becomes 'ce').
    4. Replace spaces with underscores.

    Parameters
    ----------
    columns : list of str
        A list of column names to be cleaned and formatted.
    max_length : int
        The maximum length allowed for column names after formatting.

    Returns
    -------
    list of str
        A list of cleaned and formatted column names, or None if errors occur.

    Examples
    --------
    >>> df_col(['Column 1', '(Year)', 'Another Column Example'], 3)
    ['col', 'yer', 'ano']

    Notes
    -----
    - This function uses regular expressions to clean and format column names.
    - If a column name exceeds the specified `max_length`, it is truncated using initials.
    - Any spaces in column names are replaced with underscores.
    - This function is useful for ensuring consistent column naming conventions in DataFrames.

    Raises
    ------
    IndexError
        If an index error occurs during column processing.
    re.error
        If there is a regular expression error during column processing.

    See Also
    --------
    pandas.DataFrame.rename : Pandas function for renaming DataFrame columns.
    """

extract_cordinates.__doc__=    """
    Extracts numerical coordinates from a list of coordinate strings.

    Parameters:
    -----------
    coordinates : list
        A list or Pandas Series containing coordinate strings to be processed.

    Returns:
    --------
    list
        A list where each element contains the extracted coordinates as a list of strings.
        If no coordinates are found in an element, it is represented as an empty list.

    Raises:
    -------
    ValueError
        If the input is not a list.

    Notes:
    ------
    This function uses regular expressions to extract numerical coordinates (e.g., "12.345, 67.890") from
    each element in the input list. The extracted coordinates are returned as a list of strings
    for each element. If no coordinates are found in an element, it is represented as an empty list.

    Example:
    --------
    >>> coordinates = ["12.345, 67.890", "45.678, -12.345", "invalid"]
    >>> result = extract_coordinates(coordinates)
    >>> print(result)
    [['12.345', '67.890'], ['45.678', '-12.345'], []]
    """

load_csv.__doc__=    """
    Load data from a CSV file into a Pandas DataFrame.
    
    Parameters
    ----------
    file_path : str
        The file path to the CSV file.
    
    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the loaded data if the file exists and can be loaded successfully.
        Returns None and prints an error message if the file is not found.
    
    Examples
    --------
    >>> data = load_csv('data.csv')
    >>> if data is not None:
    >>>     print(data.head())
    
    Notes
    -----
    - If the specified CSV file does not exist at the given 'file_path', an error message is printed, and None is returned.
    - Ensure that you have the necessary permissions to access the file and that the file path is correct.
    
    See Also
    --------
    pandas.read_csv : The Pandas function used internally to read CSV files.
    """

load_xlsx.__doc__=    """
    Load data from an Excel (.xlsx) file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The file path to the Excel (.xlsx) file.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the loaded data if the file exists and can be loaded successfully.
        Returns None and prints an error message if the file is not found.

    Examples
    --------
    >>> data = load_xlsx('data.xlsx')
    >>> if data is not None:
    >>>     print(data.head())

    Notes
    -----
    - If the specified Excel file does not exist at the given 'file_path', an error message is printed, and None is returned.
    - Ensure that you have the necessary permissions to access the file and that the file path is correct.

    See Also
    --------
    pandas.read_excel : The Pandas function used internally to read Excel files (.xlsx).
    """

geo_plot.__doc__=    """
    Create a geographical plot of points on a map using latitude and longitude data.

    This function takes latitude and longitude data as Pandas Series, combines them into a GeoDataFrame, and
    plots the points on a geographical map overlayed with a specified map background.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing additional data to be plotted alongside the geographical points.
    latitude : pd.Series
        A Pandas Series containing latitude values.
    longitude : pd.Series
        A Pandas Series containing longitude values.
    file_path : str
        The file path to a shapefile or GeoJSON file that defines the map background.
    title : str, optional
        An optional title for the plot.
    size : tuple, optional
        The size of the map in inches (width, height). Default is (10, 10).
    marker_size : int, optional
        The size of the markers for plotted points. Default is 10.
        
    Raises:
    -------
    TypeError:
        If latitude or longitude is not provided as Pandas Series.

    FileNotFoundError:
        If the specified file_path does not exist.

    ValueError:
        If any unexpected value-related errors occur during plotting.

    See Also:
    ---------
    - Pandas Series documentation for creating and working with Series:
      https://pandas.pydata.org/docs/reference/api/pandas.Series.html
    - GeoPandas documentation for creating GeoDataFrames and working with geographical data:
      https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geoseries

    Example:
    --------
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> data = {'City': ['New York', 'Los Angeles', 'Chicago'],
    ...         'Latitude': [40.7128, 34.0522, 41.8781],
    ...         'Longitude': [-74.0060, -118.2437, -87.6298]}
    >>> df = pd.DataFrame(data)
    >>> geo_plot(df, df['Latitude'], df['Longitude'], 'path/to/map_background.shp', title='City Locations')
    """

corr_mat.__doc__=    """
    Create and display a heatmap of the correlation matrix of numeric columns in a Pandas DataFrame.

    This function calculates the correlation matrix for the numeric columns in the given DataFrame and
    visualizes it as a heatmap with annotations.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame for which the correlation matrix will be computed and visualized.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
    >>> df = pd.DataFrame(data)
    >>> corr_mat(df)

    Notes
    -----
    - This function requires the Pandas, Seaborn, and Matplotlib libraries to be installed.
    - The correlation matrix is computed for numeric columns only, and non-numeric columns are ignored.
    - The resulting heatmap provides insights into the relationships between numeric columns.
    
    Raises
    ------
    ValueError
        If the DataFrame is empty or contains no numeric columns.
    TypeError
        If the input is not a Pandas DataFrame.
    
    See Also
    --------
    pandas.DataFrame.corr : Pandas function for calculating the correlation matrix.
    seaborn.heatmap : Seaborn function for creating a heatmap.
    """