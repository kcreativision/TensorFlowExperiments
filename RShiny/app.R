library(shiny)
library(reticulate)
library(stringi)
use_python("/usr/bin/python")
# py_run_file("trigger_tensorflowexps.py")

params_mapping = read.csv('params_mapping.csv')
withConsoleRedirect <- function(containerId, expr) {
    # Change type="output" to type="message" to catch stderr
    # (messages, warnings, and errors) instead of stdout.
    txt <- capture.output(results <- expr, type = "output")
    if (length(txt) > 0) {
        insertUI(paste0("#", containerId), where = "beforeEnd",
                 ui = paste0(txt, "\n", collapse = "")
        )
    }
    results
}

# Define UI for application that draws a histogram
ui <- fluidPage(
    includeCSS("www/styles.css"),
    titlePanel("TensorFlow Experiments"),
    tags$div(tags$h4("Dataset: FMNIST")),
    tags$div(tags$h4("Layers: Flatten, Dense-128 RELU, Dense-10 SOFTMAX")),
    tags$div(tags$h5("Change the parameters, run and visualize results")),
    width=12,
    column(12, 
           fluidRow(
               column(3, uiOutput("epochs")),
               column(3, uiOutput("optimizer")),
               column(3, uiOutput("loss_function")),
               column(3, uiOutput("batchsize")))
           ),
    
    fluidRow(column(6, actionButton("go", "SUBMIT"), 
                    style= "color: skyblue; position: releative;left:0%")),
    br(),
    fluidPage(uiOutput('train_report'))
    # fluidPage(pre(id="console"))
    # fluidRow(column(12, htmlOutput('graph')))
)


server <- function(input, output) {

    output$epochs = renderUI({
        selectInput("epochs", label="No. of epochs", choices=unique(params_mapping$epochs))
    })
    output$optimizer = renderUI({
        selectInput("optimizer", label="Choose the Optimizer", choices=unique(params_mapping$optimizer))
    })
    output$loss_function = renderUI({
        selectInput("loss_function", label="Choose Loss Function", choices=unique(params_mapping$loss_function))
    })
    output$batchsize = renderUI({
        selectInput("batchsize", label="Choose batch size", choices=unique(params_mapping$batchsize))
    })    
    
    RunAndGetText <- eventReactive(input$go, {
        optimizer_shortform <- params_mapping[which(params_mapping$epochs==input$epochs &
                                                        params_mapping$optimizer==input$optimizer &
                                                        params_mapping$loss_function==input$loss_function &
                                                        params_mapping$batchsize==input$batchsize), ]$optimizer_shortform
        loss_function_shortform <- params_mapping[which(params_mapping$epochs==input$epochs &
                                                        params_mapping$optimizer==input$optimizer &
                                                        params_mapping$loss_function==input$loss_function &
                                                        params_mapping$batchsize==input$batchsize), ]$loss_function_shortform
        source_python("trigger_tensorflowexps.py")
        
        log_dir = run_script(input$epochs, input$optimizer, optimizer_shortform, 
                   input$loss_function, loss_function_shortform, input$batchsize)
        readLines(paste('..', log_dir, 'train_report.txt', sep='/'))
    })

    
    output$train_report <- renderUI({
        rawText <- RunAndGetText()
        splitText <- stringi::stri_split(str = rawText, regex = '\\n')
        replacedText <- lapply(splitText, p)
        return(replacedText)
    })

}


shinyApp(ui = ui, server = server)
