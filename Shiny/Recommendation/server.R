

library(shiny)
library(dplyr)
library(tidyr)
library(stringr)
library(curl)
library(httr)
library(stringr)
library(bigrquery)
library(bigrquery)
githubURL <- "https://raw.githubusercontent.com/dautroc1/Recommendation-system/main/auth/recommendation-299509-bf69fd9f379c.json"
download.file(githubURL,"recommendation-299509-bf69fd9f379c.json")
bq_auth(path = "recommendation-299509-bf69fd9f379c.json",email = "nminhdang2@gmail.com")
projectid <- "recommendation-299509"



shinyServer(function(input, output) {
    
    
    
    
    
    output$text <- renderTable({
        df1 <- data.frame(message = c("Not found user id"))
        if(is.na(as.numeric(input$textinput)) == TRUE)
        {
          
          return(df1)
        }
        else if(as.numeric(input$textinput) < 0)
        {
          return(df1)
        }
        temp <- paste("https://recommendation-299509.appspot.com/recommendation?userId=",as.character(input$textinput),sep = "")
        temp1 <- paste(temp,"&numRecs=5",sep = "")
        result <- curl(temp1,"r")
        
        while(length(x <- readLines(result, n = 5))){
            
            x1 <- as.numeric(str_extract_all(x, "[0-9]+")[[1]])
            
            sql <- "SELECT * FROM bqtest.anime where anime_id IN ("
            temp2 <- paste(unlist(x1),collapse = ",")
            query1 <- paste(sql, temp2,sep = "")
            query2 <- paste(query1,")",sep = "")
            df <- bq_project_query(projectid, query2) %>% bq_table_download(page_size = 5,quiet= TRUE)
            close(result)
            
            return(df)
        }
        
        
        
        
        
    })
    
    
})
