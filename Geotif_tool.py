#!/usr/bin/env python
# coding: utf-8
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash_canvas import DashCanvas
import os
from dash_canvas.utils import (array_to_data_url, parse_jsonstring, parse_jsonstring_rectangle)
from skimage import io, color, img_as_ubyte
import io
import cv2
import numpy as np
import base64
from base64 import decodestring
from zipfile import ZipFile
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
import plotly.express as px
import json
import glob
import dash_daq as daq
import shutil

globalImage = ""

static_image_route = './static/'

if not os.path.exists(static_image_route):
    os.makedirs(static_image_route)

server = Flask(__name__)
app = dash.Dash(server=server)

@server.route("/download/<path:path>")
def download(path):
    return send_from_directory(static_image_route, path, as_attachment=True)

app.config['suppress_callback_exceptions']=True
app.config.suppress_callback_exceptions = True

canvas_width = 500

columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']

app.layout = html.Div([
        html.Div([html.H1('Options',style={'font-family':'Times New Roman', 
                                            'font-size': '25px'}),
        html.Div([
                daq.ToggleSwitch(
                    id='selectionMode',
                    label='Manual   Automatic',
                    labelPosition='bottom'
                ) 
                ],style={'margin':'5px'}), 

        html.Div(id='sideNav'),
        html.Div(id='sideNavOptions'),
        html.Div(id='GridValues'),

        html.Button('Save ROI', id='button', style={'display':'block'
                                    ,'position':'absolute',
                                    'font-size': '16px',
                                    'padding': '8px 12px',
                                    'border-radius': '4px',
                                    'text-align': 'center',
                                    'align':'center',
                                    'color':'black',
                                    'margin':'25px',
                                    'font-family':'Times New Roman',
                                    'textAlign':'center'}),
        html.Div(id='AutoDownload'),
        html.Div(id='GridDownload'),
        html.Div([
                                daq.Knob(
                                    id='my-knob',
                                    size=80,
                                    min=-3,
                                    max=3,
                                    value=0,
                                    className='dark-theme-control'
                                ),
                                html.Div(id='knob-output')
                            ],id='knob', style={'display':'None'}),


        ],style={'right':5, 'position':'fixed','top':5, 'width':'10%','height':'85%', 'background-color':'#3b6466'}),
    html.Div([
    html.Hr(),
    html.H1('NDSU Precision Ag Group'),
    html.H1('Small Grain Plots Segmentation Tool'),
    html.H4('A faster Dataset Creation Tool '),
    html.Hr(),
    html.H2('Upload Image below'),
    html.Div([
      dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select color image')
        ]),
        style={
            'width': '100%',
            'height': '300px',
            'lineHeight': '300px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'margin-bottom':'200px'
        },
        multiple=False
    ),
    html.Br(),
    
    html.Div([
    html.Div(id='output-image-upload', style={'display': 'relative',
                                                'position':'center',
                                                'align':'center',
                                                'margin-left': 'auto', 
                                                'margin-right': 'auto', 
                                                'padding-left': '40px',
                                                'padding-right': '40px',
                                                'padding-topg': '25px',
                                                'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)'}),
    html.Div(id='output-image-uploadCorrection'),
    html.Div(id='output-image-uploadGrid')

        ],style={'height':'100', 'width':'100'}),

    html.Img(id='Output-generatl_image', style={'float':'left','margin-left':'None', 'left':25,'position': 'absolute', 'left': '50%', 'margin-bottom':'20px'}),# 'transform': 'translate(-50%, 10%)'}),#,'height':'510px', 'width':'920px'}),
    html.Img(id='Output-operation_image', style={'float':'left','margin-left':'None', 'left':25,'position': 'relative', 'left': '50%','height':'510px', 'width':'920px', 'transform': 'translate(-50%, 10%)','margin-bottom':'20px'}),

          ], style={'textAlign': 'center','display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto', 'width': '32.5%','backgroundImage': 'url(https://www.pexels.com/photo/scenic-view-of-agricultural-field-against-sky-during-sunset-325944/)'}, className="five columns"),

    ], className="five columns", style={'height':'100', 'width':'100'}),
    ], style={'textAlign': 'center','background-color': 'rgb(87, 95, 110)','height':'75%','color': 'white'})


@app.callback([Output('output-image-upload', 'children'), 
               Output('upload-image','style'), 
               Output('button','style'), 
               Output('Output-generatl_image','src'), 
               Output('Output-operation_image','style'), 
               Output('knob','style')],
              [Input('upload-image', 'contents'), Input('selectionMode','value')])

def update_output_div(list_of_contents, opt):
    global globalImage
    if opt == False or opt == None:
        if list_of_contents is not None:
            MainImage = list_of_contents.encode("utf8").split(b";base64,")[1]
            IMG = io.BytesIO()
            IMG.write(base64.b64decode(MainImage))
            IMG.seek(0)
            i = np.asarray(bytearray(IMG.read()), dtype=np.uint8)
            i = cv2.imdecode(i, cv2.IMREAD_COLOR)
            i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            globalImage = i
            figu = px.imshow(i, width=920, height=510)
            figu.update_layout(dragmode="drawrect")
            figu.update_layout(coloraxis_showscale=False)
            figu.update_xaxes(showticklabels=False)
            figu.update_yaxes(showticklabels=False)
            return html.Div([
                        daq.ToggleSwitch(
                            id='Semi-Manual',
                            label='Manual and Semi-Manual crop',
                            labelPosition='bottom'
                        ),
                        dcc.Graph(id='multiAnnotation',figure=figu, 
                            config={
                                "modeBarButtonsToAdd": [
                                "drawline",
                                "drawopenpath",
                                "drawclosedpath",
                                "drawcircle",
                                "drawrect",
                                "eraseshape",
                                ]
                                }, style={'text-align': 'center', 'position': 'absolute', 'left': '50%', 'transform': 'translate(-50%, 10%)','height':'510px', 'width':'920px'}), 
                        daq.ToggleSwitch(
                            id='angleCorrection',
                            label='Correct Angle',
                            labelPosition='bottom'
                        ),
                        
        ], style={'display':'relative','left':0,'margin-right':'50px'}) , {'display':'None'}, {'display':'block'
                                                                                                ,'position':'relative',
                                                                                                'font-size': '16px',
                                                                                                'padding': '8px 12px',
                                                                                                'border-radius': '4px',
                                                                                                'text-align': 'center',
                                                                                                'align':'center',
                                                                                                'color':'black',
                                                                                                'font-family':'Times New Roman',
                                                                                                'textAlign':'center'}, None, {'display':'None'}, {'display':'block','filter':' drop-shadow(-10px 10px 4px #4a4a49)','color':'white'}
    else:
        if list_of_contents is not None:
            return None,  {'display':'None'}, {'display':'block',
                                                                'position':'relative',
                                                                'font-size': '16px',
                                                                'padding': '8px 12px',
                                                                'border-radius': '4px',
                                                                'text-align': 'center',
                                                                'align':'center',
                                                                'color':'black',
                                                                'font-family':'Times New Roman',
                                                                'textAlign':'center'}, None , {'float':'left','margin-left':'None', 'left':25,'position': 'relative', 'left': '50%','height':'510px', 'width':'920px','transform': 'translate(-50%, 10%)'},{'display':'None'}
                                                                                             
dataArray=[]
selection = 0

def generateRIOs(roiArray,i):
    count=0
    for coordinates in roiArray:
        for string in json.loads(coordinates):
            for key, value in string.items():
                x, y, x1, y1 = int(string['x0']),int(string['y0']),int(string['x1']),int(string['y1'])
        ROI = cv2.cvtColor(i[y:y1, x:x1], cv2.COLOR_BGR2RGB)
        count+=1
        cv2.imwrite(static_image_route+str(count)+'_cropped.png', cv2.cvtColor(ROI, cv2.COLOR_RGB2BGR ))
    imgs = glob.glob(static_image_route+'/'+'*.png')
    with ZipFile(static_image_route+'cropped.zip', 'w') as zipObj2:
        for image_files in imgs:
            zipObj2.write(image_files)

@app.callback(
    Output('sideNav', 'children'), #Output("annotations-data", "children"), 
    [Input("multiAnnotation", "relayoutData"),
     Input('upload-image', 'contents'), 
     Input('button', 'n_clicks')], 
    prevent_initial_call=True,
)

def on_new_annotation(relayout_data, contents, generateSeg):#, cropMode, angle, angleCor):
    
    if relayout_data is not None:
        if "shapes" in relayout_data:
            data = contents.encode("utf8").split(b";base64,")[1]
            img = io.BytesIO()
            imgNDVI = io.BytesIO()
            img.write(base64.b64decode(data))
            img.seek(0)
            i = np.asarray(bytearray(img.read()), dtype=np.uint8)
            i = cv2.imdecode(i, cv2.IMREAD_COLOR)
            if json.dumps(relayout_data["shapes"], indent=2) not in dataArray:
                dataArray.append(json.dumps(relayout_data["shapes"], indent=2))
                selection= len(dataArray)
            if generateSeg is not None:
                try:
                    generateRIOs(dataArray,i)
                except Exception: pass
                selection = len(dataArray)
                return html.A(html.Button('Download',style={'display':'block'
                                    ,'position':'relative',
                                    'top': '55%','left': '10%',
                                    'font-size': '16px',
                                    'padding': '8px 12px',
                                    'border-radius': '4px',
                                    'text-align': 'center',
                                    'align':'center',
                                    'color':'black',
                                    'font-family':'Times New Roman',
                                    'textAlign':'center'}), href=os.path.join(static_image_route,'cropped.zip'), style={'position':'relative',
                                    'top': '55%','left': '0%',})
            
            return html.Button('No of selection '+str(selection),style={'display':'block'
                                    ,'position':'relative',
                                    'top': '45%','left': '10%',
                                    'font-size': '16px',
                                    'padding': '8px 12px',
                                    'border-radius': '4px',
                                    'text-align': 'center',
                                    'align':'center',
                                    'color':'black',
                                    'font-family':'Times New Roman',
                                    'textAlign':'center'})
        else:
            return dash.no_update

def angleCalibration(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result 

@app.callback(
    [Output('output-image-uploadCorrection', 'children'),Output('output-image-upload', 'style')], #Output("annotations-data", "children"), 
    [Input('upload-image', 'contents'),
    Input('Semi-Manual','value'), 
    Input('my-knob', 'value'), 
    Input('angleCorrection', 'value')],
    prevent_initial_call=True,
)          

def ImageCaliberation(im,cropMode, angle, angleCor):
    global globalImage
    if im is not None:
        if cropMode == True:
            if angleCor is True:
                data = im.encode("utf8").split(b";base64,")[1]
                img = io.BytesIO()
                imgNDVI = io.BytesIO()
                img.write(base64.b64decode(data))
                img.seek(0)
                i = np.asarray(bytearray(img.read()), dtype=np.uint8)
                i = cv2.imdecode(i, cv2.IMREAD_COLOR)
                i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                i = angleCalibration(i, angle)
                globalImage = i 
                figu = px.imshow(i, width=920, height=510)
                figu.update_layout(dragmode="drawrect")
                figu.update_layout(margin=dict(l=0, r=0, t=0, b=0)),
                figu.update_xaxes(showticklabels=False)
                figu.update_yaxes(showticklabels=False)
                return dcc.Graph(id='multiAnnotation',figure=figu, 
                            config={
                                "modeBarButtonsToAdd": [
                                "drawline",
                                "drawopenpath",
                                "drawclosedpath",
                                "drawcircle",
                                "drawrect",
                                "eraseshape",
                                ]
                                }, style={'text-align': 'center', 
                                'position': 'absolute', 
                                'left': '50%', 
                                'transform': 'translate(-50%, 10%)',
                                'height':'510px', 'width':'920px'}), {'display':'None'}


@app.callback(
    Output('sideNavOptions', 'children'), #Output("annotations-data", "children"), 
    [Input('selectionMode','value'), Input('upload-image', 'contents')],
    prevent_initial_call=True,
)   

def automatic_seg(opt_, image):
    if opt_ == True:
        if image is not None:

            return [html.Div([
                            dcc.Input(
                                id='RLower_value',
                                type='number',
                                placeholder="RL",
                                min=0, max=255, step=1,
                            ),
                            dcc.Input(
                                id='Rupper_value',
                                type='number',
                                placeholder="RU",
                                min=0, max=255, step=1,
                            ),
                            dcc.Input(
                                id='GLower_value',
                                type='number',
                                placeholder="GL",
                                min=0, max=255, step=1,
                            ),
                            dcc.Input(
                                id='Gupper_value',
                                type='number',
                                placeholder="GU",
                                min=0, max=255, step=1,
                            ),
                            dcc.Input(
                                id='BLower_value',
                                type='number',
                                placeholder="BL",
                                min=0, max=255, step=1,
                            ),
                            dcc.Input(
                                id='Bupper_value',
                                type='number',
                                placeholder="BU",
                                min=0, max=255, step=1,
                            ),

                        ]
                        + [html.Div(id="out-all-types")]
                    ),

            daq.ToggleSwitch(
                    id='V_mask',
                    label='Apply vertical Mask',
                    labelPosition='bottom'
                ) ,
            daq.ToggleSwitch(
                    id='H_mask',
                    label='Apply horizontal Mask',
                    labelPosition='bottom'
                ) ,
            daq.ToggleSwitch(
                    id='Combined_mask',
                    label='Combine V&H Masks',
                    labelPosition='bottom'
                ) ,
            daq.ToggleSwitch(
                    id='cont',
                    label='Find Contours',
                    labelPosition='bottom'
                ) ,
                            dcc.Input(
                                id='MinArea',
                                type='number',
                                placeholder="Min Area",
                                
                            ),
                            dcc.Input(
                                id='MaxArea',
                                type='number',
                                placeholder="Max Area",
                            ),
            html.H5('Filter by Area')]

@app.callback(
    [Output('Output-operation_image', 'src'),Output('Output-generatl_image','style'),Output('AutoDownload','children')], #Output("annotations-data", "children"), 
    [Input('selectionMode','value'),
    Input('upload-image', 'contents'),
    Input('RLower_value','value'),
    Input('Rupper_value','value'),
    Input('GLower_value','value'),
    Input('Gupper_value','value'),
    Input('BLower_value','value'),
    Input('Bupper_value','value'),
    Input('V_mask','value'),
    Input('H_mask','value'),
    Input('Combined_mask','value'),
    Input('cont','value'),
    Input('MinArea','value'),
    Input('MaxArea','value'),
    Input('button', 'n_clicks')
     ],
    prevent_initial_call=True,
)  

def automatic_segOutput(opt_, image, RL, RU, GL, GU, BL, BU, V_mask, H_mask, C_mask, contour, MinA, MaxA, genRIO):
    if opt_ == True:
        if image is not None:
            MainImage = image.encode("utf8").split(b";base64,")[1]
            IMG = io.BytesIO()
            IMG.write(base64.b64decode(MainImage))
            IMG.seek(0)
            i = np.asarray(bytearray(IMG.read()), dtype=np.uint8)
            i = cv2.imdecode(i, cv2.IMREAD_COLOR)
            i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            FinalImage = i#np.zeros_like(i, np.uint8)
            mask = cv2.inRange(i, (0, 0, 0), (255, 255, 255))
            if RL is not None and RU is not None and GL  is not None and GU is not None and BL is not None and BU is not None:
                mask = cv2.inRange(i, (RL, GL, BL), (RU, GU, BU))
                imask = mask>0
                green = np.zeros_like(i, np.uint8)
                green[imask] = i[imask]
                if green is not None:
                    FinalImage = green
                    grey = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(grey,(5,5),0)
                    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    close = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=8)
                else:
                    FinalImage = i
            
                
            if V_mask == True:
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,110))
                vertical_mask = cv2.morphologyEx(close, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                vertical = cv2.dilate(vertical_mask, vertical_kernel, iterations=7)
                if vertical is not None:
                    FinalImage = vertical

            if H_mask == True:
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
                horizontal_mask = cv2.morphologyEx(close, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
                horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations=10)
                if horizontal_mask is not None:
                    FinalImage = horizontal_mask

            if C_mask == True:
                hmm = vertical==0
                horizontal_mask[hmm] = vertical[hmm]
                if horizontal_mask is not None:
                    FinalImage = horizontal_mask    

            if contour == True and MinA is not None and MaxA is not None:
                if not os.path.exists(os.path.join(static_image_route,'data')):
                    os.makedirs(os.path.join(static_image_route,'data'))
                boundingBox = horizontal_mask
                cnts = cv2.findContours(boundingBox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                image_number = 0
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > MinA and area < MaxA:
                        x,y,w,h = cv2.boundingRect(c)
                        ROI = i[y:y+h, x:x+w]
                        cv2.imwrite(static_image_route+'/data/ROI_{}.png'.format(image_number), ROI)
                        cv2.rectangle(i, (x, y), (x + w, y + h), (36,255,12), 20)
                        image_number += 1
                if boundingBox is not None:
                    FinalImage = i
                if genRIO is not None:
                    shutil.make_archive(static_image_route+'Automatic_dataset', 'zip', os.path.join(static_image_route,'data'))
                    return array_to_data_url(img_as_ubyte(FinalImage)),{'display':'None'}, html.A(html.Button('Download',style={'display':'block'
                                        ,'position':'relative',
                                        'font-size': '16px',
                                        'padding': '8px 12px',
                                        'border-radius': '4px',
                                        'text-align': 'center',
                                        'align':'center',
                                        'color':'black',
                                        'font-family':'Times New Roman',
                                        'textAlign':'center'}), href=os.path.join(static_image_route,'Automatic_dataset.zip'), style={'position':'relative',
                                        })

            return array_to_data_url(img_as_ubyte(FinalImage)),{'display':'None'}, None
        else:
            return None

@app.callback(
    [Output('GridValues','children')],
    [Input('Semi-Manual','value')],
    prevent_initial_call=True)

def inputMenu(condition):
    if condition is True:
        return [html.Div([
            dcc.Input( 
                id='verticalGrid',
                type='number',
                placeholder="Vertical Grids",
                min=0, max=100, step=1,
                    ),
            dcc.Input( 
                id='horizontalGrid',
                type='number',
                placeholder="Horizontal Grids",
                min=0, max=100, step=1,
                    ),
            dcc.Input( 
                id='subverticalGrid',
                type='number',
                placeholder="Sub-vertical Grids",
                min=0, max=10, step=1,
                    ),
            dcc.Input( 
                id='subhorizontalGrid',
                type='number',
                placeholder="Sub-horizontal Grids",
                min=0, max=10, step=1,
                    ),
            html.Button('Draw Grids', id='GridSubmit'),

                    ],style={'margin':'5px'})]

darray= []

def gridD(roiArray,i,VG, HG, SVG, SHG):
    if SHG is None:
        SHG = 1
    count=0
    for coordinates in roiArray:
        for string in json.loads(coordinates):
            for key, value in string.items():
                x, y, x1, y1 = int(string['x0']),int(string['y0']),int(string['x1']),int(string['y1'])
    ROIint = i[y:y1, x:x1]
    for x in range(0, int(ROIint.shape[1]), int(ROIint.shape[1]/VG)):
        cv2.line(ROIint,(x,0),(x,int(ROIint.shape[0])),(0,0,255),5)

    xDiv = int(int(ROIint.shape[0]/HG)/SHG)
    for x in range(0, int(ROIint.shape[0]), int(ROIint.shape[0]/HG)):
        cv2.line(ROIint,(0,x),(int(ROIint.shape[1]),x),(0,0,255),5)
        try:
            for y in range(0, x, xDiv):
                cv2.line(ROIint,(int(ROIint.shape[1]/2),y),(int(ROIint.shape[1]),y),(0,0,255),5)
        except Exception: 
            pass
        count+=1
    saveROI = True
    if saveROI == True:
        print(saveROI)
        if not os.path.exists(os.path.join(static_image_route,'dataCROP')):
            os.makedirs(os.path.join(static_image_route,'dataCROP'))
        path_ = os.path.join(static_image_route,'dataCROP')
        mask = cv2.inRange(ROIint, (0, 0, 254), (0, 0,255))
        imask = mask>0
        blue = np.zeros_like(ROIint, np.uint8)
        blue[imask] = ROIint[imask]
        greyblue = cv2.cvtColor(blue, cv2.COLOR_RGB2GRAY)
        blurblue = cv2.GaussianBlur(greyblue,(5,5),0)
        ret4,th4 = cv2.threshold(blurblue,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th4 = 255 - th4
        cnts = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        image_number = 0
        
        for c in cnts:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            ROI = ROIint[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(path_,'ROI_{}.png'.format(image_number)), ROI)
            image_number += 1
        shutil.make_archive(static_image_route+'Grid_dataset', 'zip', os.path.join(static_image_route,'dataCROP'))

    
    return ROIint, html.A(html.Button('Download',style={'display':'block'
                                        ,'position':'relative',
                                        'font-size': '16px',
                                        'padding': '8px 12px',
                                        'border-radius': '4px',
                                        'text-align': 'center',
                                        'align':'center',
                                        'color':'black',
                                        'font-family':'Times New Roman',
                                        'textAlign':'center'}), href=os.path.join(static_image_route,'Grid_dataset.zip'), style={'position':'relative',
                                        })

@app.callback(
    [Output('output-image-uploadGrid','children'),Output('output-image-uploadCorrection', 'style'),Output('GridDownload','children')], #Output("annotations-data", "children"), 
    [Input("multiAnnotation", "relayoutData"),
    Input('Semi-Manual','value'),
    Input('angleCorrection','value'),
    Input('verticalGrid','value'),
    Input('horizontalGrid','value'),
    Input('subverticalGrid','value'),
    Input('subhorizontalGrid','value'),
    ]
    ,prevent_initial_call=True,
)

def drawGrid(sel, opt1, opt2, VG, HG, SVG, SHG):
    if globalImage is not None:
        if opt1 is True:
            if opt2 is True:
                if sel is not None:
                    if json.dumps(sel["shapes"], indent=2) not in darray:
                        darray.append(json.dumps(sel["shapes"], indent=2))
                        if HG is not None and VG is not None:
                            igrph, saveButton = gridD(darray,globalImage, VG, HG, SVG, SHG)
                            figu = px.imshow(igrph, width=920, height=510)
                            figu.update_layout(dragmode="drawrect")
                            figu.update_layout(coloraxis_showscale=False)
                            figu.update_xaxes(showticklabels=False)
                            figu.update_yaxes(showticklabels=False)
                            return dcc.Graph(id='multiAnnotation2',figure=figu, 
                                    config={
                                        "modeBarButtonsToAdd": [
                                        "drawline",
                                        "drawopenpath",
                                        "drawclosedpath",
                                        "drawcircle",
                                        "drawrect",
                                        "eraseshape",
                                        ]
                                        }, style={'text-align': 'center', 
                                        'position': 'absolute', 
                                        'left': '50%', 
                                        'transform': 'translate(-50%, 10%)',
                                        'height':'510px', 'width':'920px'}), {'display':'None'}, saveButton #, html.Button('SAVE ROI', id='save_gridSeg')

############################################################################################

if __name__ == '__main__':
    app.run_server(debug=False)


