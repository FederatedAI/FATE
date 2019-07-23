package com.webank.ai.fate.board.bootstrap;


import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.ImportResource;
import org.springframework.context.annotation.PropertySource;
import org.springframework.scheduling.annotation.EnableScheduling;


@SpringBootApplication
@ComponentScan(basePackages = {"com.webank.ai.fate.board.*"})
@ImportResource(locations = {
        "classpath:db-mybatis-context.xml"
})
@PropertySource(value = "classpath:application.properties", ignoreResourceNotFound = true)
@Configuration
@EnableScheduling
public class Bootstrap {

    public static void main(String[] args) {
        try {
            ConfigurableApplicationContext context = SpringApplication.run(Bootstrap.class, args);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}



