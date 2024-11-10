# Objective: read VGLGui workflow file and load content into memory
# File type: structure.txt

#
import sys
import re
import os
import string
from collections import defaultdict
import numpy as np

lstGlyph = []                   #List to store Glyphs
lstGlyphPar = []                #List to store Glyphs Parameters
lstConnection = []              #List to store Connections
lstConnectionInput = []         #List to store Connections inputs
lstGlyphIn = []                 #List to store Glyphs Inputs
lstGlyphOut = []                #List to store Glyphs Outputs

class Error (Exception): #Class for treat a exception defined for user
    pass

# Structure for storing Glyphs in memory
# Glyph represents a function
class objGlyph(object):
        
    #Glyph:[Library]:comment::localhost:[Glyph_ID]:[Glyph_X]:[Glyph_Y]:: -[var_str] '[var_str_value]' -[var_num] [var_num_value]    
    def __init__(self, vlibrary, vfunc, vlocalhost, vglyph_id, vglyph_x, vglyph_y):       
        self.library = vlibrary                 #library name (Ex: VisionGL)
        self.func = vfunc                       #function
        self.localhost = vlocalhost             #folder where the image file or library function is located
        self.glyph_id = vglyph_id               #glyph identifier code
        self.glyph_x = vglyph_x                 #numerical coordinate of the glyph's linear position on the screen 
        self.glyph_y = vglyph_y                 #numerical coordinate of the column position of the Glyph on the screen
        
        # Rule7: Glyphs have READY (ready to run) and DONE (executed) status, both status start being FALSE
        self.ready = False                      #TRUE = glyph is ready to run
        self.done = False                       #TRUE = glyph was executed

        self.lst_par = []                       #parameter list

        # Rule1: Glyphs correspond to vertices in a graph and represent a function from the VisionGL library
        #           They have a list of input and output parameters.
        #           Each input is linked to a single output of another glyph.
        #           Each output can be connected to more than one input from another glyph.
        self.lst_input = []                     #glyph input list
        self.lst_output = []                    #glyph output list

    #Add glyph parameter function
    def funcGlyphAddPar (self, vGlyphPar):
        self.lst_par.append(vGlyphPar)

    #Add glyph input function
    def funcGlyphAddIn (self, vGlyphIn):
        self.lst_input.append(vGlyphIn)

    #Add glyph output function 
    def funcGlyphAddOut (self, vGlyphOut):
        self.lst_output.append(vGlyphOut)

    # Return Glyph Ready status
    def getGlyphReady(self):
        return self.ready

    # Rule8: Glyphs have a list of entries. When all entries are READY=TRUE, the glyph changes status to READY=TRUE (function ready to run)
    def setGlyphReady(self, status):

        vGlyphReady = status

        #Identifies if all glyph entries were used
        if vGlyphReady == True and len(self.lst_input) > 0:
            
            #If there is an entry without using
            for vGlyphIn in self.lst_input:            
                if vGlyphIn.getStatus() == False:
                    vGlyphReady = False
                    self.ready = False
                    exit    
    
            #If all inputs were used
            if vGlyphReady:
                self.ready = vGlyphReady
        else:
            self.ready = vGlyphReady

    # Rule10: Glyph becomes DONE = TRUE after its execution
    #         Assign done to glyph
    def setGlyphDone(self, status):
        self.done = status

    #Return Done status
    def getGlyphDone(self):
        return self.done

class VariableStore(object):

    def __init__(self):
        self.variables = {}  # Dicionário para armazenar variáveis

    # Adiciona ou atualiza uma variável
    def set_variable(self, name, value):
        self.variables[name] = value

    # Obtém o valor de uma variável
    def get_variable(self, name):
        return self.variables.get(name, None)

    # Avalia uma expressão usando as variáveis armazenadas
    def evaluate_expression(self, expression):
        try:
            # Usa eval para avaliar expressões, considerando as variáveis armazenadas
            return eval(expression, {}, self.variables)
        except Exception as e:
            print(f"Erro ao avaliar expressão: {expression}. Detalhes: {str(e)}")
            return None




# Structure for storing Parameters in memory
class objGlyphParameters(object):

    def __init__(self, namepar, valuepar):
        self.name = namepar      #variable name
        self.value = valuepar    #variable value
        
    def getName(self):
        return self.name

    def getValue(self):
        return self.value


# Structure for storing Glyphs input list in memory
class objGlyphInput(object):

    def __init__(self, vnamein, vstatusin):
        self.namein = vnamein            #glyph input name
        self.statusin = vstatusin        #glyph input status

    def getStatus(self):
        return self.statusin

# Structure for storing Glyphs output list in memory
class objGlyphOutput(object):

    def __init__(self, vnameout, vstatusout):
        self.nameout = vnameout      #glyph output name
        self.statusout = vstatusout  #glyph output status

    #Assign status to glyph output
    def setGlyphOutput(self, status):
        self.statusout = status

#Create the inputs and outputs for the glyph
def procCreateGlyphInOut():

    for procCreateGlyphInOut_indexConn, procCreateGlyphInOut_vConnection in enumerate(lstConnection):
        
        for procCreateGlyphInOut_i, procCreateGlyphInOut_vGlyph in enumerate(lstGlyph):

            # Create the input for the glyph
            for procCreateGlyphInOut_vInputPar in lstConnection[procCreateGlyphInOut_indexConn].lst_con_input:

                if procCreateGlyphInOut_vInputPar.Par_name != '\n' and procCreateGlyphInOut_vGlyph.glyph_id == procCreateGlyphInOut_vInputPar.Par_glyph_id:
                    procCreateGlyphInOut_vGlyphIn = objGlyphInput(procCreateGlyphInOut_vInputPar.Par_name, False)
                    lstGlyph[procCreateGlyphInOut_i].funcGlyphAddIn (procCreateGlyphInOut_vGlyphIn)

            # Create the output for the glyph   
            if procCreateGlyphInOut_vConnection.output_varname != '\n' and procCreateGlyphInOut_vGlyph.glyph_id == procCreateGlyphInOut_vConnection.output_glyph_id:
                procCreateGlyphInOut_vGlyphOut = objGlyphOutput(procCreateGlyphInOut_vConnection.output_varname, False)
                lstGlyph[procCreateGlyphInOut_i].funcGlyphAddOut (procCreateGlyphInOut_vGlyphOut)

    #Rule11: Source glyph is already created with READY = TRUE. 
    #        After creating NodeConnections, the Glyph that has no input will be considered of the SOURCE type and 
    #        will have READY = TRUE (ready for execution)
    for procCreateGlyphInOut_i, procCreateGlyphInOut_vGlyph in enumerate(lstGlyph):

       if len(procCreateGlyphInOut_vGlyph.lst_input) == 0:
           lstGlyph[procCreateGlyphInOut_i].setGlyphReady(True)

#Identifies and Creates the parameters of the Glyph
def procCreateGlyphPar(procCreateGlyphPar_vGlyph, procCreateGlyphPar_vParameters, procCreateGlyphPar_count):
    try:

        #Identifies the parameters
        #:: -[var_str] '[var_str_value]' -[var_num] [var_num_value]
        procCreateGlyphPar_contentGlyPar = []               #clears the glyph parameter list
        procCreateGlyphPar_lstParAux = []                   #auxiliary parameter list

        procCreateGlyphPar_contentGlyPar = procCreateGlyphPar_vParameters

        for procCreateGlyphPar_vpar in procCreateGlyphPar_contentGlyPar:
            if procCreateGlyphPar_vpar != '' and procCreateGlyphPar_vpar != '\n':

                procCreateGlyphPar_vGlyphPar = objGlyphParameters 
                procCreateGlyphPar_vpar = procCreateGlyphPar_vpar.replace("\n", '') 

                #Differentiates parameter name and value

                #regex codes 
                #r = re.compile(r'[^\d ]')
                #r = re.compile(r'-?\d+\.?\d*')
                #\[[\s\d\.,-]*\]

                #TODO fazer if para se for "palavra" se for "> < " 
                if procCreateGlyphPar_vpar[0] == '\'' or procCreateGlyphPar_vpar.isdigit():
                    procCreateGlyphPar_vGlyphPar = objGlyphParameters('Value', procCreateGlyphPar_vpar.replace("'", ''))

                elif procCreateGlyphPar_vpar[0].isdigit() and procCreateGlyphPar_vpar[1] == '.': #decimal number
                    procCreateGlyphPar_vpar = re.findall('-?\d+\.?\d*',procCreateGlyphPar_vpar)
                    procCreateGlyphPar_vGlyphPar = objGlyphParameters('Value', procCreateGlyphPar_vpar )

                elif procCreateGlyphPar_vpar.isdigit() or procCreateGlyphPar_vpar[0] =='[': #array
                    procCreateGlyphPar_vpar = re.findall('-?\d+\.?\d*',procCreateGlyphPar_vpar)
                    procCreateGlyphPar_vGlyphPar = objGlyphParameters('Value', procCreateGlyphPar_vpar )

                elif procCreateGlyphPar_vpar[0] == '-' and procCreateGlyphPar_vpar[1].isdigit(): #negative number
                    procCreateGlyphPar_vpar = re.findall('-?\d+\.?\d*',procCreateGlyphPar_vpar)
                    procCreateGlyphPar_vGlyphPar = objGlyphParameters('Value', procCreateGlyphPar_vpar )

                if procCreateGlyphPar_vpar[0] == "-":             
                    if procCreateGlyphPar_vpar[1].isdigit():
                        procCreateGlyphPar_vGlyphPar = objGlyphParameters('Value', procCreateGlyphPar_vpar.replace("-", ''))
                    else:
                        procCreateGlyphPar_vGlyphPar = objGlyphParameters('Name', procCreateGlyphPar_vpar.replace('-', ''))
                
                #Temporary list to differentiate parameters and their values
                procCreateGlyphPar_lstParAux.append(procCreateGlyphPar_vGlyphPar)
        
        #Creates the parameters of the Glyph
        for procCreateGlyphPar_i, procCreateGlyphPar_vParAux in enumerate(procCreateGlyphPar_lstParAux):
            
            procCreateGlyphPar_vParType = procCreateGlyphPar_vParAux.getName()
            procCreateGlyphPar_vParValue = procCreateGlyphPar_vParAux.getValue()


            procCreateGlyphPar_vParTypeNext = ''
            procCreateGlyphPar_vParValueNext = ''


            #If you don't have the next parameter to include
            if procCreateGlyphPar_i < (len(procCreateGlyphPar_lstParAux)-1):
                procCreateGlyphPar_vParTypeNext = procCreateGlyphPar_lstParAux[procCreateGlyphPar_i+1].getName()
                procCreateGlyphPar_vParValueNext = procCreateGlyphPar_lstParAux[procCreateGlyphPar_i+1].getValue()
            
                
            # A parameter name followed by another parameter name. Write the parameter because it will have no value. Example: -wh -hw -dd
            if procCreateGlyphPar_vParType == 'Name' and (procCreateGlyphPar_vParTypeNext == 'Name' or (procCreateGlyphPar_vParTypeNext == '' and procCreateGlyphPar_vParType != 'Value')):
                procCreateGlyphPar_vGlyphPar = objGlyphParameters(procCreateGlyphPar_vParValue, '')
                procCreateGlyphPar_vGlyph.funcGlyphAddPar(procCreateGlyphPar_vGlyphPar)

            # A parameter name followed by a value. Write the parameter with its value
            if procCreateGlyphPar_vParType == 'Name' and procCreateGlyphPar_vParTypeNext == 'Value':
                procCreateGlyphPar_vGlyphPar = objGlyphParameters(procCreateGlyphPar_vParValue, procCreateGlyphPar_vParValueNext)
                procCreateGlyphPar_vGlyph.funcGlyphAddPar(procCreateGlyphPar_vGlyphPar)

    except IndexError as d: #rule102 - Variable not found
        print("Non-standard information in the Parameter declaration"," \nLine",{procCreateGlyphPar_count}, "{d}")
    except ValueError as s: #rule103 - Error in defined Parameters coordinates (not integer or out of bounds)
        print("Non-standard information in the Parameter declaration","\nLine",{procCreateGlyphPar_count} , "{s}")

# Create Glyph
def procCreateGlyph(procCreateGlyph_contentGly, procCreateGlyph_count):
    try:
        
        procCreateGlyph_vBlib = ''
        procCreateGlyph_vFunc = ''
        procCreateGlyph_vLoc = ''
        procCreateGlyph_vIdGlyh = ''
        procCreateGlyph_vPosX = ''
        procCreateGlyph_vPosY = ''            
        procCreateGlyph_vGlyphPar = ''
        
        
        if len(procCreateGlyph_contentGly) == 8:  #Image Input/Outpu type Glyph
            procCreateGlyph_vBlib = procCreateGlyph_contentGly[1]
            procCreateGlyph_vFunc = procCreateGlyph_contentGly[2]
            procCreateGlyph_vLoc = procCreateGlyph_contentGly[3]
            procCreateGlyph_vIdGlyh = procCreateGlyph_contentGly[4]
            procCreateGlyph_vPosX = procCreateGlyph_contentGly[5]
            procCreateGlyph_vPosY = procCreateGlyph_contentGly[6]
            procCreateGlyph_vGlyphPar = procCreateGlyph_contentGly[7].replace(", ",',')  
            procCreateGlyph_vGlyphPar = procCreateGlyph_vGlyphPar.split(' ')  

        
        elif len(procCreateGlyph_contentGly) > 9: #Image type parameter
            procCreateGlyph_vBlib = procCreateGlyph_contentGly[1]
            procCreateGlyph_vFunc = procCreateGlyph_contentGly[2]
            procCreateGlyph_vLoc = procCreateGlyph_contentGly[4]
            procCreateGlyph_vIdGlyh = procCreateGlyph_contentGly[5]
            procCreateGlyph_vPosX = procCreateGlyph_contentGly[6]
            procCreateGlyph_vPosY = procCreateGlyph_contentGly[7]
            procCreateGlyph_vGlyphPar = procCreateGlyph_contentGly[9].replace(", ",',') 
            procCreateGlyph_vGlyphPar = procCreateGlyph_vGlyphPar.split(' ')           

        procCreateGlyph_vGlyph = objGlyph(procCreateGlyph_vBlib, procCreateGlyph_vFunc, procCreateGlyph_vLoc, procCreateGlyph_vIdGlyh, procCreateGlyph_vPosX, procCreateGlyph_vPosY)

        #Creates the parameters of the Glyph
        procCreateGlyphPar(procCreateGlyph_vGlyph, procCreateGlyph_vGlyphPar, procCreateGlyph_count)                    

        #rule104 - Invalid screen position or exceeds dimensions to be defined by file
        if (int(procCreateGlyph_contentGly[6]) or int(procCreateGlyph_contentGly[7])) > 100000 or (int(procCreateGlyph_contentGly[6]) or int(procCreateGlyph_contentGly[7])) < 0:
            raise Error("Glyph position on screen in error,", " check the line: ",{procCreateGlyph_count}) 

        #Create the Glyph
        lstGlyph.append(procCreateGlyph_vGlyph)

    except IndexError as d: #rule102 - Variable not found
        print("Non-standard information in the Glyph declaration"," \nLine",{procCreateGlyph_count}, "{d}")
    except ValueError as s: #rule103 - Error in defined glyph coordinates (not integer or out of bounds)
        print("Non-standard information in the Glyph declaration","\nLine",{procCreateGlyph_count} , "{s}")

  #Add glyph input function

# Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
#        Reading the image from another glyph does not change this status.
#        Set READY = TRUE to glyph input and READY = TRUE to glyph 
def setGlyphInputReadyByIdOut(setGlyphInputReadyByIdOut_vOutputGlyph_id):
    
    for setGlyphInputReadyByIdOut_i_Con, setGlyphInputReadyByIdOut_vConnection in enumerate(lstConnection):

        # Checks if the executed glyph is the origin of any glyph
        if setGlyphInputReadyByIdOut_vConnection.output_glyph_id == setGlyphInputReadyByIdOut_vOutputGlyph_id:

            # Assign read-ready to connection
            lstConnection[setGlyphInputReadyByIdOut_i_Con].setReadyConnection(True)

            # Search all glyph entries
            for setGlyphInputReadyByIdOut_vConnInput in setGlyphInputReadyByIdOut_vConnection.lst_con_input:
                setGlyphInputReady(setGlyphInputReadyByIdOut_vConnInput.Par_glyph_id, setGlyphInputReadyByIdOut_vConnInput.Par_name)

# Rule10: Glyph becomes DONE = TRUE after its execution. Assign done to glyph
def setGlyphDoneId(setGlyphDoneId_vGlyphIdUpd):
    for setGlyphDoneId_i_GliUpd, setGlyphDoneId_vGlyph in enumerate(lstGlyph):
        if setGlyphDoneId_vGlyph.glyph_id == setGlyphDoneId_vGlyphIdUpd:
            lstGlyph[setGlyphDoneId_i_GliUpd].setGlyphDone(True)
            break

#Assign status to glyph output
def setGlyphInputReady(setGlyphInputReady_vPar_glyph_id, setGlyphInputReady_vPar_name):

    # Finds the glyphs that originate from the executed glyph
    for setGlyphInputReady_i_Gly, setGlyphInputReady_vGlyph in enumerate(lstGlyph):

        if setGlyphInputReady_vGlyph.glyph_id == setGlyphInputReady_vPar_glyph_id:

            # Rule8: Glyphs have a list of entries. When all entries are READY=TRUE, the glyph changes status to READY=TRUE (function ready to run)
            # Set READY = TRUE to the Glyph input
            for setGlyphInputReady_i_GlyInput, setGlyphInputReady_vGlyphIn in enumerate(setGlyphInputReady_vGlyph.lst_input):           

                if setGlyphInputReady_vGlyphIn.namein == setGlyphInputReady_vPar_name:
                    lstGlyph[setGlyphInputReady_i_Gly].lst_input[setGlyphInputReady_i_GlyInput].statusin = True

            lstGlyph[setGlyphInputReady_i_Gly].setGlyphReady(True)
            break

# Structure for storing Connections in memory
# Images are stored on edges (connections between Glyphs)
class objConnection(object):

    #NodeConnection:data:[output_Glyph_ID]:[output_varname]:[input_Glyph_ID]:[input_varname]        
    def __init__(self, voutput_glyph_id, voutput_varname):       
        self.output_glyph_id = voutput_glyph_id     #glyph identifier code output
        self.output_varname = voutput_varname       #variable name output
        self.lst_con_input = []                     #glyph input list
        self.image = None                           #image
        self.ready = False                          #False = unread or unexecuted image; True = image read or executed

    #Get Image of Connection
    def getImageConnection(self):
        return self.image

    #Assign image to Connection
    def setReadyConnection(self, statusConn):
        self.ready = statusConn

    #Return if connection is ready
    def getReadyConnection(self):
        return self.ready

    #Add an entry to the input parameter list
    def addConnInput(self, vConnPar):
        self.lst_con_input.append(vConnPar)

# Structure for storing Connections input list in memory
class objConnectionPar(object):

    def __init__(self, vConnPar_id, vConnPar_Name):
        self.Par_glyph_id = vConnPar_id         #glyph identifier code Parameter
        self.Par_name = vConnPar_Name           #variable name Parameter

# Find the connection output 
def getOutputConnection(getOutputConnection_vGlyph_IdOutput):

    for getOutputConnection_vConnection in lstConnection:
        if getOutputConnection_vConnection.output_glyph_id == getOutputConnection_vGlyph_IdOutput:
            return True
            exit

    return False

# Find the connection's output of input glyph
def getOutputConnectionByIdName(getOutputConnectionByIdName_vGlyph_idInput, getOutputConnectionByIdName_vNameParInput):

    for getOutputConnectionByIdName_vConnection in lstConnection:   
        for getOutputConnectionByIdName_vInputPar in getOutputConnectionByIdName_vConnection.lst_con_input:          
            if getOutputConnectionByIdName_vInputPar.Par_glyph_id == getOutputConnectionByIdName_vGlyph_idInput and getOutputConnectionByIdName_vInputPar.Par_name == getOutputConnectionByIdName_vNameParInput:
                getOutputConnectionByIdName_vConnGet = objConnectionPar(getOutputConnectionByIdName_vConnection.output_glyph_id, getOutputConnectionByIdName_vConnection.output_varname)
                return getOutputConnectionByIdName_vConnGet

    return None

# Rule5: Each edge has an image stored
#        Assign image to Connection
def setImageConnectionByOutputId(setImageConnectionByOutputId_vGlyph_OutputId, setImageConnectionByOutputId_img):

    for setImageConnectionByOutputId_indexConn, setImageConnectionByOutputId_vConnection in enumerate(lstConnection):   
        if setImageConnectionByOutputId_vConnection.output_glyph_id == setImageConnectionByOutputId_vGlyph_OutputId:
            lstConnection[setImageConnectionByOutputId_indexConn].image = setImageConnectionByOutputId_img

# Returns edge image based on glyph id
def getImageInputByIdName(getImageInputByIdName_vGlyph_idInput, getImageInputByIdName_vNameParInput):

    for getImageInputByIdName_vConnection in lstConnection:   

        for getImageInputByIdName_vInputPar in getImageInputByIdName_vConnection.lst_con_input:          
            if getImageInputByIdName_vInputPar.Par_glyph_id == getImageInputByIdName_vGlyph_idInput and getImageInputByIdName_vInputPar.Par_name == getImageInputByIdName_vNameParInput:
                return getImageInputByIdName_vConnection.getImageConnection()

    return None

# Add the connection's input glyph
def addInputConnection (addInputConnection_vConnOutput, addInputConnection_vinput_Glyph_ID, addInputConnection_vinput_varname):

    if addInputConnection_vConnOutput is not None:
        for addInputConnection_vConnIndex, addInputConnection_vConnection in enumerate(lstConnection):   
            if addInputConnection_vConnection.output_glyph_id == addInputConnection_vConnOutput.Par_glyph_id and addInputConnection_vConnection.output_varname == addInputConnection_vConnOutput.Par_name:
                addInputConnection_vConnParIn = objConnectionPar(addInputConnection_vinput_Glyph_ID, addInputConnection_vinput_varname)
                lstConnection[addInputConnection_vConnIndex].addConnInput(addInputConnection_vConnParIn)
                break

#Creates the connections of the workflow file
def procCreateConnection(procCreateConnection_voutput_Glyph_ID, procCreateConnection_voutput_varname, procCreateConnection_vinput_Glyph_ID, procCreateConnection_vinput_varname):

    # Create the connection    
    if not getOutputConnection(procCreateConnection_voutput_Glyph_ID):
        procCreateConnection_vConnCre = objConnection(procCreateConnection_voutput_Glyph_ID, procCreateConnection_voutput_varname)
        lstConnection.append(procCreateConnection_vConnCre)

    # Create the connection input parameter
    if getOutputConnectionByIdName(procCreateConnection_vinput_Glyph_ID, procCreateConnection_vinput_varname) is None:
        procCreateConnection_vConnPar = objConnectionPar(procCreateConnection_voutput_Glyph_ID, procCreateConnection_voutput_varname)
        addInputConnection (procCreateConnection_vConnPar, procCreateConnection_vinput_Glyph_ID, procCreateConnection_vinput_varname)

# File to be read

# vfile = sys.argv[1]
vfile = 'SAMPLES/ifelse.wksp'

vGlyph = objGlyph               #Glyph in memory 
vGlyphPar = objGlyphParameters  #Glyph parameters in memory
vGlyphIn = objGlyphInput        #Glyph input in memory
vGlyphOut = objGlyphOutput      #Glyph output in memory
vConnection = objConnection     #Connection in memory
vConnectionOutput = objConnectionPar   #Connection input in memory

# Method for reading the workflow file
# Método para ler o arquivo de workflow e armazenar informações
def fileRead(lstGlyph, lstConnection, variable_store):
    try:
        if os.path.isfile(vfile):

            count = 0  # line counter
            in_variable_section = False  # Flag para identificar se estamos na seção de variáveis

            # Abre o arquivo de workflow
            with open(vfile, "r") as file1:
                for line in file1:
                    count += 1  # contador de linhas

                    # Detecta o início da seção de variáveis
                    if 'variablesbegin' in line.lower():
                        in_variable_section = True
                        continue

                    # Detecta o fim da seção de variáveis
                    if 'variablesend' in line.lower():
                        in_variable_section = False
                        continue

                    # Armazena variáveis na seção de variáveis
                    if in_variable_section:
                        if '=' in line:
                            # Extrai a variável e seu valor
                            var_name, var_value = line.split('=')
                            var_name = var_name.strip()
                            var_value = var_value.strip()

                            # Armazena no store de variáveis
                            variable_store.set_variable(var_name, int(var_value))  # Supondo valores inteiros
                            print(f"Variável '{var_name}' armazenada com valor {var_value}")

                    # Extrai os glifos
                    if ('glyph:' in line.lower()) or ('extport:' in line.lower() or 'conditional:' in line.lower()):
                        procCreateGlyph(line.split(':'), count)

                    # Cria as conexões
                    if 'nodeconnection:' in line.lower():
                        try:
                            contentCon = line.split(':')
                            voutput_Glyph_ID = contentCon[2]
                            voutput_varname = contentCon[3].replace('\n', '')
                            vinput_Glyph_ID = contentCon[4]
                            vinput_varname = contentCon[5].replace('\n', '')

                            # Verifica IDs de Glyph inválidos
                            try:
                                if int(voutput_Glyph_ID) < 0 or int(vinput_Glyph_ID) < 0:
                                    raise ValueError(f"Invalid glyph id on line: {count}")
                            except ValueError:
                                print("Invalid Connection Creation Values. Check the line:", count)

                            procCreateConnection(voutput_Glyph_ID, voutput_varname, vinput_Glyph_ID, vinput_varname)

                        except IndexError as f:
                            print(f"Connections indices not found {f} on line {count} of the file")

            # Cria inputs e outputs dos glifos
            procCreateGlyphInOut()

    except UnboundLocalError:
        print("File not found.")

variable_store = VariableStore()
fileRead(lstGlyph, lstConnection, variable_store)