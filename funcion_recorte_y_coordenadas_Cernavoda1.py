#########################################################################################################
# CREANDO EL PARQUE QUE SE RECORTARÁ
park="Cernavoda1"
wind_parks = {"Cernavoda1": {"Lat": 44.30896, "Lon": 28.16639}}   # Coordenadas de los parques de viento
crop_margin=2	# Margen empleado a la hora de recortar
region = {'lat_sup' : wind_parks[f'{park}']['Lat']+crop_margin,
          'lat_inf' : wind_parks[f'{park}']['Lat']-crop_margin,
          'lon_sup' : wind_parks[f'{park}']['Lon']+crop_margin,
          'lon_inf' : wind_parks[f'{park}']['Lon']-crop_margin}
#########################################################################################################


################################################
# Función para recortar las variables del mapa #
################################################
#####################################################
# NOTAS:
# 	- meteo_path_structure() es una función que accede al fichero donde se encuentran almacenadas las variables.
# Basta sustituirlo por la ruta hasta el fichero correspondiente.
#   - Se necesita importar la librería xarray que en esta función está importada como xr (import xarray as xr)
#   - import numpy as np
#   - 

def open_multiple_variables(region, variables, lag, path, years=None, output_format="xarray"):
    '''
    Genera una matriz de varias variables de datos

    Parameters
    ----------
    region : str
        Nombre de la region
    variables : list
        lista de variables entre las siguientes: ["wind100","wind10","cloud","temp2","radDown","radDir"]
    lag : int
        Numero correspondiente al horizonte temporal.
    output_format : str, optional
        'xarray': Devuelve los datos como un Dataset de xarray.
        'numpy': devuelve una tupla (mapas, fechas) con la matriz de mapas en formato numpy
                y un array de fechas
        Opcion por defecto 'xarray'.

    Returns
    -------
        Datos en el formato escogido

    '''

    dataset={}
    if years is None:		# DEFINIR AQUÍ LOS AÑOS DE LOS QUE SE DESEA OBTENER LAS VARIABLES
        ini=params.INIT_DATE
        fin=params.END_DATE
        years=list(range(int(ini[:4]),int(fin[:4])+1))

    for var in variables:

        # Generar nombre de archivo para la variable y region

        # Carga de mapa
        # Forma 1: leer el mapa de la region ya recortado
        if type(region) is str:
            xarray_filename=meteo_path_structured(region, var, years, lag, path, file_ext='zarr')	
            if len(xarray_filename)==0:
                xarray_filename=meteo_path_structured(region, var, years, lag, path, file_ext='nc4')
            logging.debug(f'estoy abriendo la variable {var} en el path {xarray_filename}')
            xarray_data=read_xarray(xarray_filename,output_format='xarray').load()

        # Forma 2: Leer el mapa completo y recortarlo en el momento
        elif type(region) is dict: # TODO: Cambiar a otro parametro coords=None o coords=dict
            area = region['area'] if 'area' in region else 'ESP'
            xarray_filename=meteo_path_structured(area, var, years, lag, path, file_ext='zarr')
            if len(xarray_filename)==0:
                xarray_filename=meteo_path_structured(area, var, years, lag, path, file_ext='nc4')
            logging.debug(f'estoy abriendo la variable {var} en el path {xarray_filename}')
            aux=read_xarray(xarray_filename,output_format='xarray')
            xarray_data=crop_and_load(aux, region)
            aux.close()

        if len(list(xarray_data.data_vars))==1:
            dataset[var]=xarray_data[list(xarray_data.data_vars)[0]]
            if 'reftime' in dataset[var].coords:
                dataset[var] = dataset[var].drop_vars('reftime') # Eliminamos reftime porque no casan (merge) las distintas variables
            if 'heightAboveGround' in dataset[var].coords:
                dataset[var] = dataset[var].drop_vars('heightAboveGround')
            logging.debug(f'el tamaño del dataset es {dataset[var].shape}')
            if ('cloud' in var) and dataset[var].mean().values>10:
                # TODO: cambiar mapas rumania cloud hres
                logging.warning('Division de cloud entre 100')
                dataset[var] = dataset[var]/100
        else:
            for item in list(xarray_data.data_vars):
                dataset[item]=xarray_data[item]
                if 'reftime' in dataset[item].coords:
                    dataset[item] = dataset[item].drop_vars('reftime')
                if 'heightAboveGround' in dataset[item].coords:
                    dataset[item] = dataset[item].drop_vars('heightAboveGround')

    xarray_data.close()
    # Juntamos todas las variables en un dataset. Cada variable en el dataset tiene su matriz por separado
    dataset=xr.Dataset(data_vars=dataset)

    # Juntamos las n variables en una matriz con n canales.
    # transpose() sirve para reordenar las dimensiones: primero horas,
    # luego lat y lon y por ultimo los canales
    dataset = dataset.to_array(dim='canales',name='matriz_total').transpose(...,'canales')
    dataset = dataset.dropna(dim="time")
    logging.debug('Lista de canales %s', list(dataset['canales'].values))
    if output_format=='xarray':
        return dataset
    elif output_format=='numpy':
        mapas=dataset.values
        fechas=dataset['time'].values
        return mapas, fechas
    else:
        logging.error('Formato %s no admitido. Solo se admiten los formatos "xarray" o "numpy"', output_format)



def read_xarray(xarray_filename,output_format='xarray'):
    '''
    Lee un archivo de datos con fechas y lo devuelve en formato xarray o numpy

    Parameters
    ----------
    xarray_filename : str
        Ruta al archivo
    output_format : str, optional
        'xarray': Devuelve los datos como un Dataset de xarray.
        'numpy': devuelve una tupla (mapas, fechas) con la matriz de mapas en formato numpy
                y un array de fechas
        Opcion por defecto 'xarray'.

    Returns
    -------
        Datos en el formato escogido

    '''
    logging.info('Leyendo %s', xarray_filename)

    # Para leer los archivos desde LUSTRE hay que desactivar el file locking de HDF5
    os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

    #if len(xarray_filename):
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):

    chunks={'time':1000, 'lat': 10, 'lon':10}

    if xarray_filename is str:
        ext=xarray_filename.split('.')[-1]
    else:
        ext=xarray_filename[0].split('.')[-1]

    if ext in ['grib', 'grib2']:
        engine = 'cfgrib'
    elif ext in ['nc', 'nc4']:
        engine = 'netcdf4'
    elif ext in ['zarr', 'zarr/']:
        engine = 'zarr'
        chunks = None
    else:
        engine = None

    # if (type(xarray_filename) is list) and ('s3://' in xarray_filename[0]):
    #     logging.debug('Archivos de S3')
    #     fs = s3fs.S3FileSystem()
    #     xarray_filename = [fs.get_mapper(elem.replace('\\','/')) for elem in xarray_filename]

    xarray_data = xr.open_mfdataset(xarray_filename, preprocess=fix_coords,
                                    chunks=chunks, engine=engine)

    logging.debug('Variables: %s',list(xarray_data.data_vars))
    # logging.debug('xarray leido\n %s', xarray_data)

    if output_format=='xarray':

        return xarray_data

    elif output_format=='numpy':

        fechas=xarray_data['time'].values
        logging.debug('Shape de fechas: %s',xarray_data['time'].values.shape)

        # Si hay solo una variable devolvemos la matriz numpy
        if len(list(xarray_data.data_vars))==1:
            mapas=xarray_data[list(xarray_data.data_vars)[0]].values
        # Si hay varias variables las juntamos en una unica matriz de varios canales
        else:
            xarray_data=xarray_data.to_array(dim='canales',name='matriz_total').transpose(...,'canales')
            logging.debug('Lista de canales %s', list(xarray_data['canales'].values))
            mapas=xarray_data.values
        logging.debug('Shape de mapas: %s',mapas.shape)

        return mapas, fechas
    else:
        logging.error('Formato %s no admitido. Solo se admiten los formatos "xarray" o "numpy"', output_format)



def fix_coords(ds):
    return round_coords(ds).reset_coords(drop=True)

def round_coords(ds, decimals=4):
    ds['lat'] = np.round(ds['lat'],decimals).astype('float32')
    ds['lon'] = np.round(ds['lon'],decimals).astype('float32')
    return ds


def crop_map_inner(dataset, map_edges, tol=0.001):
    lats=dataset.coords['lat']
    lons=dataset.coords['lon']
    lats_cut=lats[(lats<=map_edges['lat_sup']+tol)&(lats>=map_edges['lat_inf']-tol)]
    lons_cut=lons[(lons<=map_edges['lon_sup']+tol)&(lons>=map_edges['lon_inf']-tol)]
    return dataset.sel(dict(lat=lats_cut,lon=lons_cut))

def crop_and_load(ds, map_edges):
    return crop_map_inner(ds, map_edges).compute()