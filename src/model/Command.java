package model;

import weka.core.converters.ConverterUtils.DataSource;

public interface Command {
    void exec(DataSource trainSource, DataSource testSource);
}
