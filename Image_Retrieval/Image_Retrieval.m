function varargout = Image_Retrieval(varargin)

warning ('off','all'); 
addpath(genpath(pwd));

% IMAGE_RETRIEVAL MATLAB code for Image_Retrieval.fig
%      IMAGE_RETRIEVAL, by itself, creates a new IMAGE_RETRIEVAL or raises the existing
%      singleton*.
%
%      H = IMAGE_RETRIEVAL returns the handle to a new IMAGE_RETRIEVAL or the handle to
%      the existing singleton*.
%
%      IMAGE_RETRIEVAL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGE_RETRIEVAL.M with the given input arguments.
%
%      IMAGE_RETRIEVAL('Property','Value',...) creates a new IMAGE_RETRIEVAL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Image_Retrieval_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Image_Retrieval_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Image_Retrieval

% Last Modified by GUIDE v2.5 15-May-2017 15:39:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Image_Retrieval_OpeningFcn, ...
                   'gui_OutputFcn',  @Image_Retrieval_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Image_Retrieval is made visible.
function Image_Retrieval_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Image_Retrieval (see VARARGIN)

% Choose default command line output for Image_Retrieval
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Image_Retrieval wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Image_Retrieval_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im
[filename,pathname]=uigetfile({'*.jpg';'*.bmp';'*.png'},'select an image to search');
str=[pathname filename];
im=imread(str);
axes(handles.axes1),
imshow(im),
set(handles.edit1,'String',str),
guidata(hObject, handles);

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global index;

if strcmp(handles.popupmenu1.String{handles.popupmenu1.Value} ,'CNN (convolutional neural network)')
    load('CNN128.mat');
    load('imFiles.mat');
    index =CNN_searching_phase(CNN,handles.edit1.String);
elseif strcmp(handles.popupmenu1.String{handles.popupmenu1.Value} ,'Average color value')
    load('AvgC.mat');
    load('imFiles.mat');
    index = color_searching_phase(AvgC,handles.edit1.String );
elseif strcmp(handles.popupmenu1.String{handles.popupmenu1.Value} ,'Color Moments')
    load('cMom.mat');
    load('imFiles.mat');
    index = cMom_searching_phase(cMom,handles.edit1.String );
elseif strcmp(handles.popupmenu1.String{handles.popupmenu1.Value} ,'LBP (local binary pattern)')
    load('LBP.mat');
    load('imFiles.mat');
    index = LBP_searching_phase(LBP,handles.edit1.String );
elseif strcmp(handles.popupmenu1.String{handles.popupmenu1.Value} ,'BOF (SIFT + K-Means)')
    load('BOF.mat');
    load('imFiles.mat');
    load('clustering.mat');
    index = BOF_searching_phase(BOF,handles.edit1.String,KMeans);
end

global im2
str=imFiles{index(1)};
im2=imread(str);
axes(handles.axes3);
imshow(im2);
axes(handles.axes4);
imshow(imread(imFiles{index(2)})),
axes(handles.axes5),
imshow(imread(imFiles{index(3)}));
axes(handles.axes6),
imshow(imread(imFiles{index(4)}));
axes(handles.axes7),
imshow(imread(imFiles{index(5)}));
axes(handles.axes8),
imshow(imread(imFiles{index(6)}));
axes(handles.axes9),
imshow(imread(imFiles{index(7)}));
axes(handles.axes10),
imshow(imread(imFiles{index(8)}));
axes(handles.axes11),
imshow(imread(imFiles{index(9)}));
axes(handles.axes12),
imshow(imread(imFiles{index(10)}));

guidata(hObject, handles);
