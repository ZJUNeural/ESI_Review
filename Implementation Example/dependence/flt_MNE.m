function MNfilter = flt_MNE(L,nd,varargin)
% Minimum-norm estimation source imaging algorithms:
% S = argmin_{S} (B-LS)'C^{-1}(B-LS)+lambda ||MS||_2^{2} 
% which share the solution form of (M M')^{-1}L'(L(M M')^{-1}L'+lambda C)^{-1}; C is noise
% covariance and often taken as identity matrix in practice after whitening 
% Input:
%   L:  leadfield matrix
%   nd: number of orientations
%   lambda: regularization parameter(default:1/(9^2))
%   MM: prior of source covariance(which is (M M'),default:identity matrix)
%   Covy: noise covariance(default:identity matrix)
%   VertConn: adjacency matrix of source space(required by e.g., loreta)
%   VertLoc:  coordinates of source dipoles
%   type: MN methods(default:mne, others include:sloreta,eloreta,dspm,loreta,laura)

%% Initialize
[nChan,nSource] = size(L);
nSource = nSource/nd;
lambda = (1/9)^2*trace(L*L')/nChan;
MM = 'auto';
Covy = eye(nChan);
VertConn = [];
VertLoc = [];
type = 'mne';

if mod(length(varargin),2)==1
    error('Incorrect optional input');
else
    for ni = 1:2:length(varargin)
        Para = lower(varargin{ni});
        Value = varargin{ni+1};
        switch Para
            case 'lambda'
                lambda = Value;
            case 'mm'
                MM = Value;
            case 'covy'
                Covy = Value;
            case 'vertconn'
                VertConn = Value;
            case 'vertloc'
                VertLoc = Value;
            case 'type'
                type = lower(Value);
        end       
    end
end

%% Whitening noise covariance
[Uy,Sy] = eig((Covy+Covy.')/2);
Sn = sqrt(diag(Sy)); Uy = Uy(:,1:rank(Covy)); Sn = Sn(1:rank(Covy));
iW = diag(1./Sn)*Uy.';
L = iW*L;
%% Construct Source Priors
if strcmp(MM,'auto')
    MM = ConstructSourcePrior(L,nd,nSource,type,VertConn,VertLoc);    
end
%% MNE Filter
% inverse of gram matrix: L(MM')^{-1}L'+lambda C
[UL,SL,~] = svd(L/MM*L.');
SL = diag(SL);
iG = UL*diag(1./(SL+lambda))*UL.';
switch type
    case 'sloreta'
        [MNfilter,~] = sloreta_filt(L,MM,iG,nd);
        MNfilter = MNfilter*iW;
    case 'eloreta'
        [MNfilter,~] = eloreta_filt(L,MM,iG,lambda,nd,30);
        MNfilter = MNfilter*iW;
    case 'dspm'
        Kernel = MM\L'*iG;
        R = sum(Kernel.^2,2);
        if nd>1
            R = sum(reshape(R,nd,[]),1);
            R =  repmat(R,[nd,1]);
            R = R(:);
        end
        MNfilter = bsxfun(@rdivide,Kernel,sqrt(R))*iW;    
    otherwise
        MNfilter = MM\L'*iG*iW;
end
end

function MM = ConstructSourcePrior(L,nd,nSource,type,VertConn,VertLoc)
normL = sum(L.^2,1);
if nd==1
    normL = sqrt(normL);
    Lbound = max(normL)./100;
    W = normL;
    W(W<Lbound) = Lbound;
end
if nd>1
    normL = reshape(normL,nd,[]);
    normL = squeeze(sum(normL,1));
    normL = sqrt(normL);
    Lbound = max(normL)./100;
    W = normL;
    W(W<Lbound) = Lbound;
end
switch type
    case 'mne'
        MM = sparse(diag(W));  % depth bias compensation based on the norm of leadfield matrix      
    case 'loreta'
        A = diag(sum(VertConn,2))-0.99*VertConn; % the distance between dipoles are ignored here
        AA = A.'*A;
%         tW = repmat(W,nSource,1);
        tW = spdiags(W.^2,0,speye(nSource,nSource));
        MM = sparse(tW'.*(AA+1e-4*trace(AA)/nSource*eye(nSource)).*tW);
    case 'laura'
        dist = zeros(nSource);
        for iv = 1:nSource
            dist(iv,:) = sqrt(sum((VertLoc - repmat(VertLoc(iv,:),nSource,1)).^2,2))';
        end
        A = VertConn.*(dist.^(-2)); A(isinf(A)) = 0;
        A = diag(sum(A,2))-A;
        AA = A.'*A;
        tW = repmat(M,nSource,1);
        MM = sparse(tW'.*(AA+1e-4*trace(AA)/nSource*eye(nSource)).*tW);
    otherwise % no depth compensation is required by post-hoc normalization approaches
        MM = speye(nSource);
end
MM = kron(MM,eye(nd));
end

function [W,nL] = sloreta_filt(L,Covs,iG,nd)
W = Covs\L'*iG;

if nd==1   
    W = bsxfun(@rdivide,W,sqrt(sum(W.*L',2)));
    nL = bsxfun(@times,L',(sum(W.*L',2))).';
    return
end
if nd>1
    nL = L;
    for iter = 1:nd:size(W,1)
        curIdx = iter:iter+nd-1;
        R = W(curIdx,:)*L(:,curIdx);
        W(curIdx,:) = sqrtm(pinv(R))*W(curIdx,:);
        nL(:,curIdx) = nL(:,curIdx)*sqrtm((R));
    end
    return
end
end

function [Wout,nL] = eloreta_filt(L,Covs,iG,lamda,nd,maxIter)
nchan = size(L,1);
nc = size(L,2)/nd;
nL = L;
Wold = repmat(eye(nd),1,nc);
Wold = reshape(Wold,nd,nd,[]);
W = Wold;
Winv = W;
M = iG;
for iter = 1:maxIter
    for ic = 1:nc
        curIdx = (1+(ic-1)*nd):ic*nd;
        W(:,:,ic) = sqrtm(L(:,curIdx)'*M*L(:,curIdx));  
        Winv(:,:,ic) = pinv(W(:,:,ic));
        
    end
    winvk = zeros(size(L'));
    for ic = 1:nc
       curIdx = (1+(ic-1)*nd):ic*nd;
       winvk(curIdx,:) = Winv(:,:,ic)*L(:,curIdx)';
       
    end
    kwinvk = L*winvk;
    M = pinv(kwinvk+lamda*trace(kwinvk)*eye(nchan)/nchan); 
    if norm(reshape(W-Wold,[],1))/norm(reshape(Wold,[],1))<1e-6
        break;
    end
    Wold = W;   
end
for ic = 1:nc
    nL(:,curIdx) = nL(:,curIdx)*W(:,:,ic);
end
Wout = winvk*M;
end