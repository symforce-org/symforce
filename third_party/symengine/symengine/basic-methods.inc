class EvalRealDoubleVisitorFinal;

#define SYMENGINE_INCLUDE_METHODS_BASE()                                       \
    virtual void accept(Visitor &v) const = 0;                                 \
    virtual void accept(EvalRealDoubleVisitorFinal &v) const = 0;

#define SYMENGINE_INCLUDE_METHODS_DERIVED()                                    \
    void accept(Visitor &v) const override;                                    \
    void accept(EvalRealDoubleVisitorFinal &v) const override;
