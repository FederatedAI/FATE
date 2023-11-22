package org.fedai.osx.broker.http;

import java.io.IOException;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;

public class EmbeddedAsyncServer
{
    public static class EmbeddedAsyncServlet extends HttpServlet
    {
        @Override
        protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException
        {
            final AsyncContext ctxt = req.startAsync();
            ctxt.start(new Runnable()
            {
                @Override
                public void run()
                {

                    System.err.println("In AsyncContext / Start / Runnable / run");
                    ctxt.complete();
                }
            });
        }
    }

    public static void main(String[] args) throws Exception
    {
        Server server = new Server(9090);
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        ServletHolder asyncHolder = context.addServlet(EmbeddedAsyncServlet.class,"/async");
        asyncHolder.setAsyncSupported(true);
        server.setHandler(context);
        server.start();
        server.join();
    }
}