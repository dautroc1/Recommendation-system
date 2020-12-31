shinyUI(fluidPage(
    
    # Application title
    titlePanel("Anime recommendation"),
    
    # Sidebar with a slider input for number of bins
    sidebarLayout(
        sidebarPanel(
            h2("Instruction"),
            h5("1. Input user id to the box"),
            h5("2. Press enter and the recommendation will display below.")
            
            
        ),
        # Show a plot of the generated distribution
        mainPanel(
            tabsetPanel(
                tabPanel("Recommendation",
                         textInput("textinput","Input value"),
                         submitButton("Enter"),
                         tableOutput("text")))
            
        )
    )
))
