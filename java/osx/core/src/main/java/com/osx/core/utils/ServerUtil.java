
package com.osx.core.utils;

import java.util.Properties;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class ServerUtil {

    public static Options buildCommandlineOptions(final Options options) {
        Option opt = new Option("h", "help", false, "Print help");
        opt.setRequired(false);
        options.addOption(opt);

        opt =
            new Option("n", "namesrvAddr", true,
                "Name server address list, eg: 192.168.0.1:9876;192.168.0.2:9876");
        opt.setRequired(false);
        options.addOption(opt);

        return options;
    }

    public static CommandLine parseCmdLine(final String appName, String[] args, Options options,
        CommandLineParser parser) {
        HelpFormatter hf = new HelpFormatter();
        hf.setWidth(110);
        CommandLine commandLine = null;
        try {
            commandLine = parser.parse(options, args);
            if (commandLine.hasOption('h')) {
                hf.printHelp(appName, options, true);
                System.exit(0);
            }
        } catch (ParseException e) {
            hf.printHelp(appName, options, true);
            e.printStackTrace();
            System.exit(1);
        }

        return commandLine;
    }

    public static void printCommandLineHelp(final String appName, final Options options) {
        HelpFormatter hf = new HelpFormatter();
        hf.setWidth(110);
        hf.printHelp(appName, options, true);
    }

    public static Properties commandLine2Properties(final CommandLine commandLine) {
        Properties properties = new Properties();
        Option[] opts = commandLine.getOptions();

        if (opts != null) {
            for (Option opt : opts) {
                String name = opt.getLongOpt();
                String value = commandLine.getOptionValue(name);
                if (value != null) {
                    properties.setProperty(name, value);
                }
            }
        }

        return properties;
    }

}
